import cv2
import os
import argparse
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoExtractor:
    def __init__(self, input_folder: str, output_folder: str, videos_dict: Dict[str, List[str]]):
        """
        Initialize video extractor
        :param input_folder: Input folder containing video files
        :param output_folder: Output folder for extracted video segments
        :param videos_dict: Dictionary containing video filenames and time ranges
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.videos_dict = videos_dict
        
        # Create output directory if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Supported video formats
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    def parse_time(self, time_str: str) -> Optional[int]:
        """
        Parse time string and convert to seconds
        Supports formats: HH:MM:SS, MM:SS, or SS
        """
        try:
            parts = time_str.split(':')
            if len(parts) == 3:  # HH:MM:SS
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:  # MM:SS
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 1:  # SS
                return int(parts[0])
            else:
                raise ValueError(f"Invalid time format: {time_str}")
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing time '{time_str}': {e}")
            return None

    def seconds_to_frame(self, seconds: int, fps: float) -> int:
        """Convert seconds to frame number"""
        return int(seconds * fps)

    def validate_video_file(self, video_path: Path) -> bool:
        """Validate if the video file exists and has supported format"""
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False
        
        if video_path.suffix.lower() not in self.supported_formats:
            logger.warning(f"Unsupported video format: {video_path.suffix}")
            return False
        
        return True

    def parse_time_range(self, time_range: str) -> Optional[Tuple[int, int]]:
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

    def get_video_info(self, video_cap: cv2.VideoCapture) -> Tuple[float, int, int, int]:
        """Get video information: fps, total_frames, width, height"""
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return fps, total_frames, width, height

    def extract_single_segment(self, video_cap: cv2.VideoCapture, start_frame: int, 
                             end_frame: int, output_path: Path, fps: float, 
                             width: int, height: int) -> bool:
        """Extract a single video segment"""
        try:
            # Set video position to start frame
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Create video writer with better codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error(f"Failed to create video writer for {output_path}")
                return False
            
            frames_to_extract = end_frame - start_frame
            frames_extracted = 0
            
            while frames_extracted < frames_to_extract:
                ret, frame = video_cap.read()
                if not ret:
                    logger.warning(f"Could not read frame {start_frame + frames_extracted}")
                    break
                
                out.write(frame)
                frames_extracted += 1
            
            out.release()
            
            if frames_extracted < frames_to_extract:
                logger.warning(f"Only extracted {frames_extracted}/{frames_to_extract} frames for {output_path}")
            
            logger.info(f"✅ Segment saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting segment to {output_path}: {e}")
            return False

    def extract_video_segments(self) -> Dict[str, int]:
        """
        Extract video segments based on the videos dictionary
        Returns a summary of successful extractions
        """
        summary = {"successful": 0, "failed": 0, "skipped": 0}
        
        for video_name, time_ranges in self.videos_dict.items():
            video_path = self.input_folder / video_name
            
            # Validate video file
            if not self.validate_video_file(video_path):
                summary["skipped"] += len(time_ranges)
                continue
            
            # Open video file
            video_cap = cv2.VideoCapture(str(video_path))
            if not video_cap.isOpened():
                logger.error(f"Unable to open video: {video_name}")
                summary["skipped"] += len(time_ranges)
                continue
            
            try:
                # Get video information
                fps, total_frames, width, height = self.get_video_info(video_cap)
                total_duration = total_frames / fps if fps > 0 else 0
                
                logger.info(f"Processing video: {video_name} (Duration: {total_duration:.2f}s, FPS: {fps:.2f})")
                
                # Create output subdirectory for this video
                video_name_stem = video_path.stem
                segment_folder = self.output_folder / video_name_stem
                segment_folder.mkdir(exist_ok=True)
                
                # Process each time range
                for i, time_range in enumerate(time_ranges):
                    parsed_range = self.parse_time_range(time_range)
                    if parsed_range is None:
                        summary["failed"] += 1
                        continue
                    
                    start_seconds, end_seconds = parsed_range
                    
                    # Convert to frames
                    start_frame = self.seconds_to_frame(start_seconds, fps)
                    end_frame = self.seconds_to_frame(end_seconds, fps)
                    
                    # Validate frame range
                    if start_frame >= total_frames:
                        logger.warning(f"Start time exceeds video duration for {video_name}: {time_range}")
                        summary["failed"] += 1
                        continue
                    
                    if end_frame > total_frames:
                        logger.warning(f"End time exceeds video duration, using video end: {time_range}")
                        end_frame = total_frames
                    
                    # Generate output filename
                    start_str = time_range.split('-')[0].replace(':', '')
                    end_str = time_range.split('-')[1].replace(':', '')
                    segment_filename = f"{video_name_stem}_segment_{i+1:03d}_{start_str}_{end_str}.mp4"
                    output_path = segment_folder / segment_filename
                    
                    # Extract segment
                    if self.extract_single_segment(video_cap, start_frame, end_frame, 
                                                 output_path, fps, width, height):
                        summary["successful"] += 1
                    else:
                        summary["failed"] += 1
            
            finally:
                video_cap.release()
        
        return summary

    def print_summary(self, summary: Dict[str, int]):
        """Print extraction summary"""
        total = sum(summary.values())
        logger.info(f"\n{'='*50}")
        logger.info(f"EXTRACTION SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total segments: {total}")
        logger.info(f"✅ Successful: {summary['successful']}")
        logger.info(f"❌ Failed: {summary['failed']}")
        logger.info(f"⏭️  Skipped: {summary['skipped']}")
        logger.info(f"{'='*50}")


def load_videos_dict(json_file: Optional[str] = None, json_string: Optional[str] = None) -> Dict[str, List[str]]:
    """Load videos dictionary from JSON file or string"""
    try:
        if json_file:
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif json_string:
            return json.loads(json_string)
        else:
            # Default example
            return {
                "1.mp4": ["00:00:10-00:00:20"]
            }
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading videos dictionary: {e}")
        return {}


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract segments from videos based on timestamps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python video_extractor.py input_videos output_segments
            python video_extractor.py input_videos output_segments --json-file config.json
            python video_extractor.py input_videos output_segments --json-string '{"video.mp4": ["00:01:00-00:02:00"]}'
        """
    )
    
    parser.add_argument('input_folder', type=str, 
                       help="Folder containing input video files")
    parser.add_argument('output_folder', type=str, 
                       help="Folder to store extracted video segments")
    
    # JSON input options (mutually exclusive)
    json_group = parser.add_mutually_exclusive_group()
    json_group.add_argument('--json-file', type=str, 
                           help="JSON file containing videos dictionary")
    json_group.add_argument('--json-string', type=str, 
                           help="JSON string containing videos dictionary")
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help="Set logging level")
    
    return parser.parse_args()


def main():
    """Main function"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Load videos dictionary
        videos_dict = load_videos_dict(args.json_file, args.json_string)
        
        if not videos_dict:
            logger.error("No valid videos dictionary provided. Exiting.")
            return 1
        
        logger.info(f"Loaded configuration for {len(videos_dict)} video(s)")
        
        # Create VideoExtractor instance and extract segments
        extractor = VideoExtractor(args.input_folder, args.output_folder, videos_dict)
        summary = extractor.extract_video_segments()
        
        # Print summary
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

# Basic usage with default configuration
# python video_extractor.py input_videos output_segments

# Using JSON file
# python video_extractor.py input_videos output_segments --json-file config.json

# Using JSON string
# python video_extractor.py input_videos output_segments --json-string '{"video.mp4": ["00:01:00-00:02:00", "00:05:00-00:06:00"]}'

# With debug logging
# python video_extractor.py input_videos output_segments --log-level DEBUG