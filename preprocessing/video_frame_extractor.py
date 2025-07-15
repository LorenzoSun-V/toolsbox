import argparse
import os
import cv2
import logging
from time import time
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


class VideoFrameExtractor:
    """
    A high-performance video frame extractor with multi-threading support.
    Extracts frames from video files and saves them as JPEG images.
    """
    
    # Supported video formats
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')
    
    def __init__(self, root_path: str, out_path: str, skip_frame: int = 5, 
                 jpg_quality: int = 80, workers: Optional[int] = None):
        """
        Initialize the VideoFrameExtractor.
        
        Args:
            root_path: Input directory containing video files
            out_path: Output directory for extracted frames
            skip_frame: Number of frames to skip between extractions
            jpg_quality: JPEG compression quality (1-100)
            workers: Number of worker threads (defaults to CPU count)
        """
        self.root_path = Path(root_path)
        self.out_path = Path(out_path)
        self.skip_frame = max(1, skip_frame)  # Ensure at least 1
        self.jpg_quality = max(1, min(100, jpg_quality))  # Clamp to 1-100
        self.workers = workers or min(multiprocessing.cpu_count(), 8)
        
        # Setup logging
        self._setup_logging()
        
        # Validate paths
        self._validate_paths()
        
        # Statistics
        self.total_videos = 0
        self.processed_videos = 0
        self.total_frames_extracted = 0

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('video_extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _validate_paths(self):
        """Validate input and output paths."""
        if not self.root_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.root_path}")
        
        if not self.root_path.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {self.root_path}")
        
        # Create output directory if it doesn't exist
        self.out_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Input directory: {self.root_path}")
        self.logger.info(f"Output directory: {self.out_path}")

    def _get_video_files(self, recursive: bool = True) -> List[Path]:
        """
        Get list of video files from the input directory.
        
        Args:
            recursive: If True, search subdirectories recursively
            
        Returns:
            List of video file paths
        """
        video_files = []
        
        if recursive:
            # Recursively search all subdirectories
            for file_path in self.root_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.VIDEO_EXTENSIONS:
                    video_files.append(file_path)
        else:
            # Only search the root directory (current behavior)
            for file_path in self.root_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.VIDEO_EXTENSIONS:
                    video_files.append(file_path)
        
        self.logger.info(f"Found {len(video_files)} video files{'(recursive)' if recursive else ''}")
        return video_files

    def _extract_frames_from_video(self, video_path: Path, output_dir: Path) -> int:
        """
        Extract frames from a single video file.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames
            
        Returns:
            Number of frames extracted
        """
        try:
            # Open video capture
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"Failed to open video: {video_path}")
                return 0
            
            # Get video properties for better file naming
            fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if unable to get FPS
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_count = 0
            saved_count = 0
            start_time = int(time() * 1000)
            
            # JPEG encoding parameters
            jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality, 
                          cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame based on skip_frame interval
                if frame_count % self.skip_frame == 0:
                    # Generate timestamp-based filename
                    timestamp = start_time + int((frame_count / fps) * 1000)
                    filename = f"{timestamp:013d}_{saved_count:06d}.jpg"
                    frame_path = output_dir / filename
                    
                    # Save frame with error handling
                    success = cv2.imwrite(str(frame_path), frame, jpeg_params)
                    if success:
                        saved_count += 1
                    else:
                        self.logger.warning(f"Failed to save frame: {frame_path}")
                
                frame_count += 1
            
            cap.release()
            
            self.logger.info(f"Extracted {saved_count} frames from {video_path.name}")
            return saved_count
            
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {str(e)}")
            return 0

    def _process_single_video(self, video_path: Path) -> Tuple[str, int]:
        """
        Process a single video file (wrapper for threading).
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (video_name, frames_extracted)
        """
        try:
            # Create output directory for this video
            video_name = video_path.stem
            output_dir = self.out_path / video_name
            output_dir.mkdir(exist_ok=True)
            
            # Skip if directory already contains files (optional optimization)
            if any(output_dir.iterdir()):
                self.logger.info(f"Output directory for {video_name} already exists and contains files, skipping...")
                return video_name, 0
            
            frames_extracted = self._extract_frames_from_video(video_path, output_dir)
            return video_name, frames_extracted
            
        except Exception as e:
            self.logger.error(f"Error in thread processing {video_path}: {str(e)}")
            return video_path.name, 0

    def extract_all_frames(self) -> dict:
        """
        Extract frames from all video files using multi-threading.
        
        Returns:
            Dictionary with extraction statistics
        """
        video_files = self._get_video_files()
        if not video_files:
            self.logger.warning("No video files found in the input directory")
            return {"total_videos": 0, "processed_videos": 0, "total_frames": 0}
        
        self.total_videos = len(video_files)
        self.logger.info(f"Starting extraction with {self.workers} workers")
        
        start_time = time()
        
        # Process videos with thread pool
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Submit all tasks
            future_to_video = {
                executor.submit(self._process_single_video, video_path): video_path
                for video_path in video_files
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(video_files), desc="Processing Videos", unit="video") as pbar:
                for future in as_completed(future_to_video):
                    video_path = future_to_video[future]
                    try:
                        video_name, frames_extracted = future.result()
                        self.processed_videos += 1
                        self.total_frames_extracted += frames_extracted
                        pbar.set_postfix({
                            'Current': video_name[:20],
                            'Frames': frames_extracted
                        })
                    except Exception as e:
                        self.logger.error(f"Failed to process {video_path}: {str(e)}")
                    finally:
                        pbar.update(1)
        
        end_time = time()
        processing_time = end_time - start_time
        
        # Log final statistics
        stats = {
            "total_videos": self.total_videos,
            "processed_videos": self.processed_videos,
            "total_frames": self.total_frames_extracted,
            "processing_time": processing_time,
            "videos_per_second": self.processed_videos / processing_time if processing_time > 0 else 0
        }
        
        self.logger.info("=" * 50)
        self.logger.info("EXTRACTION COMPLETE")
        self.logger.info(f"Total videos found: {stats['total_videos']}")
        self.logger.info(f"Successfully processed: {stats['processed_videos']}")
        self.logger.info(f"Total frames extracted: {stats['total_frames']}")
        self.logger.info(f"Processing time: {stats['processing_time']:.2f} seconds")
        self.logger.info(f"Average: {stats['videos_per_second']:.2f} videos/second")
        self.logger.info("=" * 50)
        
        return stats


def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract frames from video files with multi-threading support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'root_path',
        help='Input directory containing video files'
    )
    
    parser.add_argument(
        'out_path',
        help='Output directory for extracted frames'
    )
    
    parser.add_argument(
        '--skip',
        type=int,
        default=5,
        help='Number of frames to skip between extractions (higher = fewer frames)'
    )
    
    parser.add_argument(
        '--jpg_quality',
        type=int,
        default=80,
        help='JPEG quality (1-100, higher = better quality, larger files)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker threads (default: auto-detect based on CPU cores)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Create and run extractor
        extractor = VideoFrameExtractor(
            root_path=args.root_path,
            out_path=args.out_path,
            skip_frame=args.skip,
            jpg_quality=args.jpg_quality,
            workers=args.workers
        )
        
        stats = extractor.extract_all_frames()
        
        # Exit with appropriate code
        if stats['processed_videos'] == 0:
            exit(1)  # No videos processed
        elif stats['processed_videos'] < stats['total_videos']:
            exit(2)  # Some videos failed
        else:
            exit(0)  # All videos processed successfully
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        exit(130)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()