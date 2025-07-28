import cv2
import os
import argparse
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import subprocess
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoExtractor:
    def __init__(self, input_folder: str, output_folder: str, videos_dict: Dict[str, List[str]], 
                 validate_output: bool = True):
        """
        Initialize video extractor
        :param input_folder: Input folder containing video files
        :param output_folder: Output folder for extracted video segments
        :param videos_dict: Dictionary containing video filenames and time ranges
        :param validate_output: Whether to validate output quality
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.videos_dict = videos_dict
        self.validate_output = validate_output
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    def parse_time(self, time_str: str) -> Optional[float]:
        try:
            parts = time_str.split(':')
            if len(parts) == 3: return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2: return float(parts[0]) * 60 + float(parts[1])
            if len(parts) == 1: return float(parts[0])
            raise ValueError(f"Invalid time format: {time_str}")
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing time '{time_str}': {e}")
            return None

    def validate_video_file(self, video_path: Path) -> bool:
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False
        if video_path.suffix.lower() not in self.supported_formats:
            logger.warning(f"Unsupported video format: {video_path.suffix}")
            return False
        return True

    def parse_time_range(self, time_range: str) -> Optional[Tuple[float, float]]:
        try:
            start_time_str, end_time_str = time_range.split('-')
            start_seconds = self.parse_time(start_time_str.strip())
            end_seconds = self.parse_time(end_time_str.strip())
            if start_seconds is None or end_seconds is None: return None
            if start_seconds >= end_seconds:
                logger.warning(f"Invalid time range: {time_range} (start >= end)")
                return None
            return start_seconds, end_seconds
        except ValueError as e:
            logger.error(f"Error parsing time range '{time_range}': {e}")
            return None

    def check_ffmpeg_available(self) -> bool:
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_video_info(self, video_path: Path) -> Dict[str, any]:
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'stream=codec_name,codec_type,width,height,duration,r_frame_rate:format=duration', '-of', 'json', str(video_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                video_info = {'duration': None, 'video_codec': None, 'audio_codec': None, 'width': None, 'height': None, 'fps': None, 'has_audio': False}
                if 'format' in data and 'duration' in data['format']: video_info['duration'] = float(data['format']['duration'])
                for stream in data.get('streams', []):
                    if stream['codec_type'] == 'video':
                        video_info['video_codec'] = stream.get('codec_name')
                        video_info['width'] = stream.get('width')
                        video_info['height'] = stream.get('height')
                        if 'duration' in stream and video_info['duration'] is None: video_info['duration'] = float(stream['duration'])
                        if 'r_frame_rate' in stream and '/' in stream['r_frame_rate']:
                            num, den = stream['r_frame_rate'].split('/')
                            if float(den) != 0: video_info['fps'] = float(num) / float(den)
                    elif stream['codec_type'] == 'audio':
                        video_info['has_audio'] = True
                        video_info['audio_codec'] = stream.get('codec_name')
                return video_info
        except Exception as e:
            logger.error(f"Error getting video info for {video_path}: {e}")
        return {}

    def extract_with_ffmpeg_robust(self, input_path: Path, output_path: Path, 
                                 start_time: float, end_time: float,
                                 video_info: Dict) -> bool:
        """
        Extracts a video segment using a single, robust ffmpeg command optimized for ACCURACY and compatibility.
        """
        duration = end_time - start_time
        
        # This command uses "output seeking" (-ss after -i) to guarantee frame-accurate cuts.
        # It is slower but necessary for problematic source files.
        cmd = [
            'ffmpeg', '-y', '-hide_banner',
            '-i', str(input_path),        # Input file FIRST
            '-ss', str(start_time),       # THEN specify start time for accurate seeking
            '-t', str(duration),          # Duration of the clip to extract
            
            # --- Video settings for maximum compatibility ---
            '-c:v', 'libx264',            # H.264 is the most widely supported video codec.
            '-preset', 'medium',          # A good balance between encoding speed and file size.
            '-crf', '23',                 # Standard quality factor. Lower is higher quality. 23 is a good default.
            '-pix_fmt', 'yuv420p',        # Force the most compatible pixel format.
            '-profile:v', 'main',         # A common H.264 profile for broad compatibility.
            
            # --- Audio settings ---
        ]
        if video_info.get('has_audio'):
            logger.debug(f"Audio detected (codec: {video_info.get('audio_codec')}). Re-encoding to AAC.")
            cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
        else:
            cmd.extend(['-an'])
        
        # --- Container and other flags ---
        cmd.extend([
            '-movflags', '+faststart',    # Essential for web streaming.
            str(output_path)
        ])

        logger.debug(f"Executing ACCURACY-optimized command (output seeking): {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"✅ Segment saved (accurate re-encode): {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Extraction failed for {output_path}.")
            logger.error(f"FFmpeg command: {' '.join(e.cmd)}")
            logger.error(f"FFmpeg stderr:\n{e.stderr}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during extraction: {e}")
            return False

    def validate_video_segment(self, video_path: Path) -> Dict[str, any]:
        try:
            if not video_path.exists() or video_path.stat().st_size < 1024: return {"valid": False, "reason": "File too small or doesn't exist"}
            video_info = self.get_video_info(video_path)
            if not video_info.get('width') or not video_info.get('height'): return {"valid": False, "reason": "Invalid video dimensions"}
            if video_info.get('duration', 0) <= 0.1: return {"valid": False, "reason": "Duration too short"}
            return {"valid": True, "properties": video_info}
        except Exception as e:
            logger.error(f"Error validating video {video_path}: {e}")
            return {"valid": False, "reason": f"Validation error: {e}"}

    def extract_video_segments(self) -> Dict[str, int]:
        summary = {"successful": 0, "failed": 0, "skipped": 0, "validated": 0, "validation_failed": 0}
        if not self.check_ffmpeg_available():
            logger.error("FFmpeg is required but not found. Please install FFmpeg.")
            return summary
        
        for video_name, time_ranges in self.videos_dict.items():
            video_path = self.input_folder / video_name
            if not self.validate_video_file(video_path):
                summary["skipped"] += len(time_ranges)
                continue
            
            video_info = self.get_video_info(video_path)
            if not video_info.get('duration'):
                logger.error(f"Could not get duration for {video_name}"); continue
            
            segment_folder = self.output_folder / video_path.stem
            segment_folder.mkdir(exist_ok=True)
            logger.info(f"Processing video: {video_name} (Duration: {video_info['duration']:.2f}s, Codec: {video_info.get('video_codec', 'N/A')}, Audio: {video_info.get('audio_codec', 'N/A')})")
            
            for i, time_range in enumerate(time_ranges):
                parsed_range = self.parse_time_range(time_range)
                if parsed_range is None:
                    summary["failed"] += 1; continue
                
                start_seconds, end_seconds = parsed_range
                if start_seconds >= video_info['duration']:
                    logger.warning(f"Start time {start_seconds:.2f}s exceeds video duration. Skipping.")
                    summary["skipped"] += 1; continue
                end_seconds = min(end_seconds, video_info['duration'])
                
                start_str, end_str = time_range.replace(':', '').replace('.', '').split('-')
                segment_filename = f"{video_path.stem}_segment_{i+1:03d}_{start_str}_{end_str}.mp4"
                output_path = segment_folder / segment_filename
                
                if self.extract_with_ffmpeg_robust(video_path, output_path, start_seconds, end_seconds, video_info):
                    if self.validate_output:
                        validation = self.validate_video_segment(output_path)
                        if validation["valid"]:
                            summary["validated"] += 1; summary["successful"] += 1
                        else:
                            logger.warning(f"Validation failed for {output_path}: {validation['reason']}")
                            summary["validation_failed"] += 1; summary["failed"] += 1
                    else:
                        summary["successful"] += 1
                else:
                    summary["failed"] += 1
        return summary

    def print_summary(self, summary: Dict[str, int]):
        total = summary['successful'] + summary['failed'] + summary['skipped']
        if total == 0: logger.info("No segments were processed."); return
        logger.info(f"\n{'='*60}\nEXTRACTION SUMMARY\n{'='*60}")
        logger.info(f"Total Segments: {total}")
        logger.info(f"✅ Successful: {summary['successful']}")
        logger.info(f"❌ Failed: {summary['failed']}")
        logger.info(f"⏭️  Skipped: {summary['skipped']}")
        if self.validate_output:
            logger.info(f"✓ Validated: {summary['validated']}")
            logger.info(f"✗ Validation Failed: {summary['validation_failed']}")
        logger.info(f"{'='*60}")

def load_videos_dict(json_file: Optional[str] = None, json_string: Optional[str] = None) -> Dict[str, List[str]]:
    try:
        if json_file:
            with open(json_file, 'r', encoding='utf-8') as f: return json.load(f)
        if json_string: return json.loads(json_string)
        logger.warning("No JSON source provided. Using default example.")
        return {"1.mp4": ["00:00:10-00:00:12", "00:00:15-00:00:17"]}
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading videos dictionary: {e}")
        return {}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extracts video segments with a focus on accuracy and compatibility.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_folder', type=str, help="Folder containing input video files.")
    parser.add_argument('output_folder', type=str, help="Folder to store extracted video segments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--json-file', type=str, help="JSON file with video names and time ranges.")
    group.add_argument('--json-string', type=str, help="JSON string with video names and time ranges.")
    parser.add_argument('--no-validate', action='store_true', help="Disable validation of output video segments.")
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help="Set logging level (default: INFO).")
    return parser.parse_args()

def main():
    try:
        args = parse_arguments()
        logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
        
        videos_dict = load_videos_dict(args.json_file, args.json_string)
        if not videos_dict:
            logger.error("No valid videos dictionary loaded. Exiting."); return 1
        
        logger.info(f"Loaded configuration for {len(videos_dict)} video(s).")
        extractor = VideoExtractor(args.input_folder, args.output_folder, videos_dict, validate_output=not args.no_validate)
        summary = extractor.extract_video_segments()
        extractor.print_summary(summary)
        return 0 if summary["successful"] > 0 else 1
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user."); return 1
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}", exc_info=True); return 1

if __name__ == "__main__":
    exit(main())