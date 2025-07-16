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
                 align_keyframes: bool = True, validate_output: bool = True, keyframe_interval: float = 1.0):
        """
        Initialize video extractor
        :param input_folder: Input folder containing video files
        :param output_folder: Output folder for extracted video segments
        :param videos_dict: Dictionary containing video filenames and time ranges
        :param align_keyframes: Whether to align cuts to keyframes
        :param validate_output: Whether to validate output quality
        :param keyframe_interval: Assumed keyframe interval in seconds if detection fails
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.videos_dict = videos_dict
        self.align_keyframes = align_keyframes
        self.validate_output = validate_output
        self.keyframe_interval = keyframe_interval
        
        # Create output directory if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Supported video formats
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    def parse_time(self, time_str: str) -> Optional[float]:
        """Parse time string and convert to seconds (float for precision)"""
        try:
            parts = time_str.split(':')
            if len(parts) == 3:  # HH:MM:SS
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:  # MM:SS
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 1:  # SS
                return float(parts[0])
            else:
                raise ValueError(f"Invalid time format: {time_str}")
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing time '{time_str}': {e}")
            return None

    def validate_video_file(self, video_path: Path) -> bool:
        """Validate if the video file exists and has supported format"""
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False
        
        if video_path.suffix.lower() not in self.supported_formats:
            logger.warning(f"Unsupported video format: {video_path.suffix}")
            return False
        
        return True

    def parse_time_range(self, time_range: str) -> Optional[Tuple[float, float]]:
        """Parse time range string and return start and end times in seconds"""
        try:
            start_time_str, end_time_str = time_range.split('-')
            start_seconds = self.parse_time(start_time_str.strip())
            end_seconds = self.parse_time(end_time_str.strip())
            
            if start_seconds is None or end_seconds is None:
                return None
            
            if start_seconds >= end_seconds:
                logger.warning(f"Invalid time range: {time_range} (start >= end)")
                return None
            
            return start_seconds, end_seconds
        except ValueError as e:
            logger.error(f"Error parsing time range '{time_range}': {e}")
            return None

    def check_ffmpeg_available(self) -> bool:
        """Check if ffmpeg is available on the system"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_video_info(self, video_path: Path) -> Dict[str, any]:
        """Get comprehensive video information"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'stream=codec_name,codec_type,width,height,duration,r_frame_rate:format=duration',
                '-of', 'json', str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                video_info = {
                    'duration': None,
                    'video_codec': None,
                    'audio_codec': None,
                    'width': None,
                    'height': None,
                    'fps': None,
                    'has_audio': False
                }
                
                # Get format duration
                if 'format' in data and 'duration' in data['format']:
                    video_info['duration'] = float(data['format']['duration'])
                
                # Analyze streams
                for stream in data.get('streams', []):
                    if stream['codec_type'] == 'video':
                        video_info['video_codec'] = stream.get('codec_name')
                        video_info['width'] = stream.get('width')
                        video_info['height'] = stream.get('height')
                        if 'duration' in stream and video_info['duration'] is None:
                            video_info['duration'] = float(stream['duration'])
                        
                        # Parse frame rate
                        if 'r_frame_rate' in stream:
                            fps_str = stream['r_frame_rate']
                            if '/' in fps_str:
                                num, den = fps_str.split('/')
                                video_info['fps'] = float(num) / float(den)
                    
                    elif stream['codec_type'] == 'audio':
                        video_info['has_audio'] = True
                        video_info['audio_codec'] = stream.get('codec_name')
                
                return video_info
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
        
        return {}

    def get_keyframes_fast(self, video_path: Path, max_duration: float = 60.0) -> List[float]:
        """Fast keyframe detection with timeout for short analysis"""
        try:
            # Only analyze first part of long videos to speed up
            read_duration = min(max_duration, 30.0)
            
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'packet=pts_time,flags',
                '-select_streams', 'v:0', '-of', 'csv=p=0', '-show_packets',
                '-read_intervals', f'%+#{read_duration}',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                keyframes = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            pts_time_str, flags = parts[0], parts[1]
                            if 'K' in flags and pts_time_str != 'N/A':
                                try:
                                    timestamp = float(pts_time_str)
                                    keyframes.append(timestamp)
                                except ValueError:
                                    continue
                
                if keyframes:
                    logger.info(f"Fast method: Found {len(keyframes)} keyframes")
                    return sorted(keyframes)
        
        except Exception as e:
            logger.debug(f"Fast keyframe detection failed: {e}")
        
        return []

    def generate_smart_keyframes(self, duration: float, fps: float = 25.0) -> List[float]:
        """Generate smart keyframes based on video characteristics"""
        keyframes = [0.0]  # Always start with 0
        
        # For short videos, use smaller intervals
        if duration <= 10:
            interval = 0.5  # Every 0.5 seconds for very short videos
        elif duration <= 30:
            interval = 1.0  # Every 1 second for short videos
        else:
            interval = self.keyframe_interval  # Use configured interval
        
        current_time = interval
        while current_time < duration:
            keyframes.append(current_time)
            current_time += interval
        
        logger.info(f"Generated {len(keyframes)} smart keyframes (interval: {interval}s)")
        return keyframes

    def find_safe_keyframe(self, keyframes: List[float], target_time: float, 
                          duration: float, search_before: bool = True) -> float:
        """Find safe keyframe considering video duration"""
        if not keyframes:
            return max(0.0, min(target_time, duration))
        
        if search_before:
            # Find the last keyframe before or at target time
            suitable_keyframes = [kf for kf in keyframes if kf <= target_time]
            if suitable_keyframes:
                return max(suitable_keyframes)
            else:
                return 0.0  # Use start if no suitable keyframe found
        else:
            # Find the first keyframe after or at target time
            suitable_keyframes = [kf for kf in keyframes if kf >= target_time]
            if suitable_keyframes:
                return min(suitable_keyframes)
            else:
                return duration  # Use end if no suitable keyframe found

    def extract_with_ffmpeg_robust(self, input_path: Path, output_path: Path, 
                                 start_time: float, end_time: float,
                                 video_info: Dict, keyframes: List[float]) -> bool:
        """Robust extraction handling various codec issues"""
        
        duration = end_time - start_time
        
        # Strategy selection based on video characteristics
        strategies = []
        
        # Strategy 1: For very short segments, force re-encoding with padding
        if duration <= 2.0:
            strategies.append({
                'name': 'short_segment_reencoding',
                'start': max(0, start_time - 1.0),  # Start 1 second earlier
                'input_duration': min(duration + 2.0, video_info.get('duration', duration + 2.0)),
                'output_start': 1.0 if start_time >= 1.0 else start_time,
                'output_duration': duration,
                'cmd_extra': [
                    '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                    '-c:a', 'aac', '-b:a', '128k',
                    '-ss', str(1.0 if start_time >= 1.0 else start_time),
                    '-t', str(duration)
                ]
            })
        
        # Strategy 2: Keyframe-aligned with proper audio handling
        if self.align_keyframes and keyframes and duration > 2.0:
            aligned_start = self.find_safe_keyframe(keyframes, start_time, 
                                                   video_info.get('duration', end_time), True)
            aligned_end = self.find_safe_keyframe(keyframes, end_time, 
                                                 video_info.get('duration', end_time), False)
            
            if aligned_end > aligned_start:
                if video_info.get('has_audio') and video_info.get('audio_codec') == 'pcm_alaw':
                    # Handle problematic audio codec
                    strategies.append({
                        'name': 'keyframe_aligned_audio_convert',
                        'start': aligned_start,
                        'input_duration': aligned_end - aligned_start,
                        'cmd_extra': [
                            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k'
                        ]
                    })
                else:
                    # Standard keyframe-aligned copy
                    strategies.append({
                        'name': 'keyframe_aligned_copy',
                        'start': aligned_start,
                        'input_duration': aligned_end - aligned_start,
                        'cmd_extra': ['-c', 'copy']
                    })
        
        # Strategy 3: Precise cut with re-encoding
        if video_info.get('has_audio') and video_info.get('audio_codec') == 'pcm_alaw':
            strategies.append({
                'name': 'precise_with_audio_convert',
                'start': start_time,
                'input_duration': duration,
                'cmd_extra': [
                    '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
                    '-c:a', 'aac', '-b:a', '128k'
                ]
            })
        else:
            strategies.append({
                'name': 'precise_high_quality',
                'start': start_time,
                'input_duration': duration,
                'cmd_extra': [
                    '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
                    '-c:a', 'aac', '-b:a', '128k'
                ]
            })
        
        # Strategy 4: Video-only extraction (fallback)
        strategies.append({
            'name': 'video_only',
            'start': start_time,
            'input_duration': duration,
            'cmd_extra': [
                '-c:v', 'libx264', '-crf', '20', '-preset', 'fast',
                '-an'  # No audio
            ]
        })
        
        # Try each strategy
        for strategy in strategies:
            try:
                logger.debug(f"Trying strategy: {strategy['name']}")
                
                if 'output_start' in strategy:
                    # Two-pass extraction for short segments
                    temp_path = output_path.with_suffix('.temp.mp4')
                    
                    # First pass: extract larger segment
                    cmd1 = [
                        'ffmpeg', '-y',
                        '-ss', str(strategy['start']),
                        '-i', str(input_path),
                        '-t', str(strategy['input_duration']),
                        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                        '-c:a', 'aac', '-b:a', '128k',
                        '-avoid_negative_ts', 'make_zero',
                        str(temp_path)
                    ]
                    
                    result1 = subprocess.run(cmd1, capture_output=True, text=True)
                    if result1.returncode != 0:
                        continue
                    
                    # Second pass: extract precise segment
                    cmd2 = [
                        'ffmpeg', '-y',
                        '-ss', str(strategy['output_start']),
                        '-i', str(temp_path),
                        '-t', str(strategy['output_duration']),
                        '-c', 'copy',
                        str(output_path)
                    ]
                    
                    result2 = subprocess.run(cmd2, capture_output=True, text=True)
                    
                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()
                    
                    if result2.returncode == 0:
                        logger.info(f"✅ Segment saved ({strategy['name']}): {output_path}")
                        return True
                    
                else:
                    # Single-pass extraction
                    cmd = [
                        'ffmpeg', '-y',
                        '-ss', str(strategy['start']),
                        '-i', str(input_path),
                        '-t', str(strategy['input_duration']),
                    ] + strategy['cmd_extra'] + [
                        '-avoid_negative_ts', 'make_zero',
                        '-fflags', '+genpts',
                        str(output_path)
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info(f"✅ Segment saved ({strategy['name']}): {output_path}")
                        return True
                    else:
                        logger.debug(f"Strategy {strategy['name']} failed: {result.stderr}")
                        
            except Exception as e:
                logger.debug(f"Strategy {strategy['name']} error: {e}")
                continue
        
        logger.error(f"All extraction strategies failed for {output_path}")
        return False

    def validate_video_segment(self, video_path: Path) -> Dict[str, any]:
        """Enhanced video validation with gray frame detection"""
        try:
            # Basic file checks
            if not video_path.exists() or video_path.stat().st_size < 1024:
                return {"valid": False, "reason": "File too small or doesn't exist"}
            
            # Get video info
            video_info = self.get_video_info(video_path)
            
            if not video_info.get('width') or not video_info.get('height'):
                return {"valid": False, "reason": "Invalid video dimensions"}
            
            if video_info.get('duration', 0) <= 0:
                return {"valid": False, "reason": "Invalid duration"}
            
            # For very short videos, do additional checks
            if video_info.get('duration', 0) <= 3.0:
                # Check if we can actually read frames
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        cap.release()
                        return {"valid": False, "reason": "Cannot open video file"}
                    
                    # Read first few frames to check for gray frames
                    frame_count = 0
                    valid_frames = 0
                    
                    for _ in range(min(10, int(video_info.get('fps', 25)))):  # Check first second
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        
                        # Simple gray frame detection
                        if len(frame.shape) == 3:
                            # Check if frame has reasonable variance (not all gray)
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
                            if variance > 10:  # Threshold for non-gray frames
                                valid_frames += 1
                    
                    cap.release()
                    
                    if frame_count > 0 and valid_frames / frame_count < 0.5:
                        return {"valid": False, "reason": f"Too many gray frames ({valid_frames}/{frame_count})"}
                    
                except Exception as e:
                    return {"valid": False, "reason": f"Frame analysis failed: {e}"}
            
            return {
                "valid": True, 
                "properties": video_info
            }
            
        except Exception as e:
            logger.error(f"Error validating video {video_path}: {e}")
            return {"valid": False, "reason": f"Validation error: {e}"}

    def extract_video_segments(self) -> Dict[str, int]:
        """Extract video segments with enhanced error handling"""
        summary = {"successful": 0, "failed": 0, "skipped": 0, "validated": 0, "validation_failed": 0}
        
        if not self.check_ffmpeg_available():
            logger.error("FFmpeg is required but not found. Please install FFmpeg.")
            return summary
        
        logger.info(f"Video extraction mode: {'Smart keyframe-aligned' if self.align_keyframes else 'Precise cut'}")
        logger.info(f"Output validation: {'Enabled' if self.validate_output else 'Disabled'}")
        
        for video_name, time_ranges in self.videos_dict.items():
            video_path = self.input_folder / video_name
            
            # Validate video file
            if not self.validate_video_file(video_path):
                summary["skipped"] += len(time_ranges)
                continue
            
            # Get comprehensive video info
            video_info = self.get_video_info(video_path)
            if not video_info.get('duration'):
                logger.error(f"Could not get duration for {video_name}")
                summary["skipped"] += len(time_ranges)
                continue
            
            # Get keyframes
            keyframes = []
            if self.align_keyframes:
                keyframes = self.get_keyframes_fast(video_path, video_info['duration'])
                if not keyframes:
                    keyframes = self.generate_smart_keyframes(
                        video_info['duration'], 
                        video_info.get('fps', 25.0)
                    )
            
            # Create output subdirectory
            video_name_stem = video_path.stem
            segment_folder = self.output_folder / video_name_stem
            segment_folder.mkdir(exist_ok=True)
            
            logger.info(f"Processing video: {video_name} (Duration: {video_info['duration']:.2f}s, "
                       f"Codec: {video_info.get('video_codec', 'unknown')}, "
                       f"Audio: {video_info.get('audio_codec', 'none') if video_info.get('has_audio') else 'none'})")
            
            # Process each time range
            for i, time_range in enumerate(time_ranges):
                parsed_range = self.parse_time_range(time_range)
                if parsed_range is None:
                    summary["failed"] += 1
                    continue
                
                start_seconds, end_seconds = parsed_range
                
                # Validate time range
                if start_seconds >= video_info['duration']:
                    logger.warning(f"Start time exceeds video duration for {video_name}: {time_range}")
                    summary["failed"] += 1
                    continue
                
                if end_seconds > video_info['duration']:
                    logger.warning(f"End time exceeds video duration, adjusting: {time_range}")
                    end_seconds = video_info['duration']
                
                # Generate output filename
                start_str = time_range.split('-')[0].replace(':', '').replace('.', '')
                end_str = time_range.split('-')[1].replace(':', '').replace('.', '')
                segment_filename = f"{video_name_stem}_segment_{i+1:03d}_{start_str}_{end_str}.mp4"
                output_path = segment_folder / segment_filename
                
                # Extract segment
                if self.extract_with_ffmpeg_robust(video_path, output_path, 
                                                 start_seconds, end_seconds, video_info, keyframes):
                    # Validate output if requested
                    if self.validate_output:
                        validation = self.validate_video_segment(output_path)
                        if validation["valid"]:
                            summary["validated"] += 1
                            summary["successful"] += 1
                        else:
                            logger.warning(f"Validation failed for {output_path}: {validation['reason']}")
                            summary["validation_failed"] += 1
                            summary["failed"] += 1
                    else:
                        summary["successful"] += 1
                else:
                    summary["failed"] += 1
        
        return summary

    def print_summary(self, summary: Dict[str, int]):
        """Print extraction summary"""
        total = sum(summary.values())
        logger.info(f"\n{'='*60}")
        logger.info(f"EXTRACTION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total segments: {total}")
        logger.info(f"✅ Successful: {summary['successful']}")
        logger.info(f"❌ Failed: {summary['failed']}")
        logger.info(f"⏭️  Skipped: {summary['skipped']}")
        if self.validate_output:
            logger.info(f"✓ Validated: {summary['validated']}")
            logger.info(f"✗ Validation failed: {summary['validation_failed']}")
        logger.info(f"{'='*60}")


def load_videos_dict(json_file: Optional[str] = None, json_string: Optional[str] = None) -> Dict[str, List[str]]:
    """Load videos dictionary from JSON file or string"""
    try:
        if json_file:
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif json_string:
            return json.loads(json_string)
        else:
            return {"1.mp4": ["00:00:10-00:00:20"]}
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading videos dictionary: {e}")
        return {}


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract segments from videos without gray frames (final version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python video_extractor.py input_videos output_segments --json-file config.json
            python video_extractor.py input_videos output_segments --precise-cut
            python video_extractor.py input_videos output_segments --keyframe-interval 0.5
        """
    )
    
    parser.add_argument('input_folder', type=str, help="Folder containing input video files")
    parser.add_argument('output_folder', type=str, help="Folder to store extracted video segments")
    
    json_group = parser.add_mutually_exclusive_group()
    json_group.add_argument('--json-file', type=str, help="JSON file containing videos dictionary")
    json_group.add_argument('--json-string', type=str, help="JSON string containing videos dictionary")
    
    parser.add_argument('--precise-cut', action='store_true', help="Use precise cutting")
    parser.add_argument('--no-validate', action='store_true', help="Skip output validation")
    parser.add_argument('--keyframe-interval', type=float, default=1.0, help="Keyframe interval (default: 1.0s)")
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help="Set logging level")
    
    return parser.parse_args()


def main():
    """Main function"""
    try:
        args = parse_arguments()
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        videos_dict = load_videos_dict(args.json_file, args.json_string)
        
        if not videos_dict:
            logger.error("No valid videos dictionary provided. Exiting.")
            return 1
        
        logger.info(f"Loaded configuration for {len(videos_dict)} video(s)")
        
        extractor = VideoExtractor(
            args.input_folder, 
            args.output_folder, 
            videos_dict,
            align_keyframes=not args.precise_cut,
            validate_output=not args.no_validate,
            keyframe_interval=args.keyframe_interval
        )
        summary = extractor.extract_video_segments()
        
        extractor.print_summary(summary)
        
        return 0 if summary["failed"] == 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())