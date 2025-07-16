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
import numpy as np
import subprocess
import tempfile
import json


class VideoFrameExtractor:
    """
    A high-performance video frame extractor with FFmpeg and OpenCV support.
    Uses FFmpeg for HEVC videos to avoid gray frame issues.
    """
    
    # Supported video formats
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')
    
    # HEVC codecs that need FFmpeg processing
    HEVC_CODECS = {'hevc', 'h265', 'x265'}
    
    def __init__(self, root_path: str, out_path: str, skip_frame: int = 5, 
                 jpg_quality: int = 80, workers: Optional[int] = None,
                 use_ffmpeg: bool = True, validate_frames: bool = True):
        """
        Initialize the VideoFrameExtractor.
        
        Args:
            root_path: Input directory containing video files
            out_path: Output directory for extracted frames
            skip_frame: Number of frames to skip between extractions
            jpg_quality: JPEG compression quality (1-100)
            workers: Number of worker threads (defaults to CPU count)
            use_ffmpeg: Use FFmpeg for HEVC videos
            validate_frames: Validate frame quality before saving
        """
        self.root_path = Path(root_path)
        self.out_path = Path(out_path)
        self.skip_frame = max(1, skip_frame)
        self.jpg_quality = max(1, min(100, jpg_quality))
        self.workers = workers or min(multiprocessing.cpu_count(), 8)
        self.use_ffmpeg = use_ffmpeg
        self.validate_frames = validate_frames
        
        # Setup logging
        self._setup_logging()
        
        # Check FFmpeg availability
        self.ffmpeg_available = self._check_ffmpeg()
        
        # Validate paths
        self._validate_paths()
        
        # Statistics
        self.total_videos = 0
        self.processed_videos = 0
        self.total_frames_extracted = 0
        self.skipped_corrupted_frames = 0
        self.ffmpeg_processed = 0

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

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.logger.info("FFmpeg detected and available")
                return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        self.logger.warning("FFmpeg not available - will use OpenCV only")
        return False

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
        """Get list of video files from the input directory."""
        video_files = []
        
        if recursive:
            for file_path in self.root_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.VIDEO_EXTENSIONS:
                    video_files.append(file_path)
        else:
            for file_path in self.root_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.VIDEO_EXTENSIONS:
                    video_files.append(file_path)
        
        self.logger.info(f"Found {len(video_files)} video files{'(recursive)' if recursive else ''}")
        return video_files

    def _get_video_info(self, video_path: Path) -> dict:
        """Get video information using FFprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'stream=codec_name,width,height,duration,r_frame_rate',
                '-of', 'json', str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                video_info = {
                    'codec': None,
                    'width': None,
                    'height': None,
                    'duration': None,
                    'fps': 25.0,
                    'is_hevc': False
                }
                
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video' or 'codec_name' in stream:
                        codec = stream.get('codec_name', '').lower()
                        video_info['codec'] = codec
                        video_info['is_hevc'] = codec in self.HEVC_CODECS
                        video_info['width'] = stream.get('width')
                        video_info['height'] = stream.get('height')
                        
                        # Get duration
                        if 'duration' in stream:
                            video_info['duration'] = float(stream['duration'])
                        
                        # Parse frame rate
                        if 'r_frame_rate' in stream:
                            fps_str = stream['r_frame_rate']
                            if '/' in fps_str and fps_str != '0/0':
                                try:
                                    num, den = fps_str.split('/')
                                    video_info['fps'] = float(num) / float(den)
                                except (ValueError, ZeroDivisionError):
                                    pass
                        break
                
                return video_info
                
        except Exception as e:
            self.logger.debug(f"FFprobe failed for {video_path}: {e}")
        
        # Fallback to OpenCV
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else None
                cap.release()
                
                return {
                    'codec': 'unknown',
                    'width': width,
                    'height': height,
                    'duration': duration,
                    'fps': fps,
                    'is_hevc': False
                }
        except Exception:
            pass
        
        return {'codec': 'unknown', 'is_hevc': False, 'fps': 25.0}

    def _is_frame_valid(self, frame: np.ndarray) -> bool:
        """Check if a frame is valid (not corrupted/gray)."""
        if frame is None or frame.size == 0:
            return False
        
        # Check if frame is mostly gray/corrupted
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Calculate variance and mean
        variance = np.var(gray)
        mean_val = np.mean(gray)
        
        # Thresholds for validation
        min_variance = 50.0
        
        # Reject frames that are too uniform
        if variance < min_variance:
            return False
        
        # Check for completely black or white frames
        if mean_val < 5 or mean_val > 250:
            return False
        
        # Check for reasonable pixel distribution
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        max_bin = np.max(hist)
        total_pixels = gray.shape[0] * gray.shape[1]
        
        # If more than 90% of pixels are the same value, likely corrupted
        if max_bin > 0.9 * total_pixels:
            return False
        
        return True

    def _extract_frames_with_ffmpeg(self, video_path: Path, output_dir: Path, 
                                   video_info: dict) -> Tuple[int, int]:
        """Extract frames using FFmpeg to avoid HEVC issues."""
        try:
            fps = video_info.get('fps', 25.0)
            
            # Calculate frame extraction rate
            extract_fps = fps / self.skip_frame
            
            # Create temporary directory for raw frames
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # FFmpeg command to extract frames
                cmd = [
                    'ffmpeg', '-y', '-i', str(video_path),
                    '-vf', f'fps={extract_fps}',
                    '-q:v', '2',  # High quality
                    '-f', 'image2',
                    str(temp_path / 'frame_%08d.jpg')
                ]
                
                # Run FFmpeg with timeout
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode != 0:
                    self.logger.error(f"FFmpeg extraction failed for {video_path.name}: {result.stderr}")
                    return 0, 0
                
                # Process extracted frames
                extracted_files = sorted(temp_path.glob('frame_*.jpg'))
                saved_count = 0
                skipped_count = 0
                start_time = int(time() * 1000)
                
                for i, frame_file in enumerate(extracted_files):
                    try:
                        # Read frame for validation
                        frame = cv2.imread(str(frame_file))
                        
                        # Validate frame if enabled
                        if self.validate_frames and not self._is_frame_valid(frame):
                            skipped_count += 1
                            continue
                        
                        # Generate output filename
                        timestamp = start_time + int((i / extract_fps) * 1000)
                        filename = f"{timestamp:013d}_{saved_count:06d}.jpg"
                        output_path = output_dir / filename
                        
                        # Copy/convert frame with desired quality
                        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality, 
                                     cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                        
                        success = cv2.imwrite(str(output_path), frame, jpeg_params)
                        if success:
                            saved_count += 1
                        else:
                            skipped_count += 1
                            
                    except Exception as e:
                        self.logger.debug(f"Error processing frame {frame_file}: {e}")
                        skipped_count += 1
                
                return saved_count, skipped_count
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"FFmpeg timeout for {video_path.name}")
            return 0, 0
        except Exception as e:
            self.logger.error(f"FFmpeg extraction error for {video_path.name}: {e}")
            return 0, 0

    def _extract_frames_with_opencv(self, video_path: Path, output_dir: Path, 
                                   video_info: dict) -> Tuple[int, int]:
        """Extract frames using OpenCV (fallback method)."""
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"Failed to open video with OpenCV: {video_path}")
                return 0, 0
            
            # Set buffer size to reduce memory usage
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            fps = video_info.get('fps', 25.0)
            frame_count = 0
            saved_count = 0
            skipped_count = 0
            start_time = int(time() * 1000)
            
            # JPEG encoding parameters
            jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality, 
                          cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            
            # Track consecutive failures
            consecutive_failures = 0
            max_consecutive_failures = 50
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > max_consecutive_failures:
                        self.logger.warning(f"Too many consecutive failures for {video_path.name}")
                        break
                    continue
                
                consecutive_failures = 0
                
                # Extract frame based on skip_frame interval
                if frame_count % self.skip_frame == 0:
                    # Validate frame if enabled
                    if self.validate_frames and not self._is_frame_valid(frame):
                        skipped_count += 1
                        self.logger.debug(f"Skipped corrupted frame {frame_count} from {video_path.name}")
                    else:
                        # Generate timestamp-based filename
                        timestamp = start_time + int((frame_count / fps) * 1000)
                        filename = f"{timestamp:013d}_{saved_count:06d}.jpg"
                        frame_path = output_dir / filename
                        
                        # Save frame
                        success = cv2.imwrite(str(frame_path), frame, jpeg_params)
                        if success:
                            saved_count += 1
                        else:
                            skipped_count += 1
                
                frame_count += 1
            
            return saved_count, skipped_count
            
        except Exception as e:
            self.logger.error(f"OpenCV extraction error for {video_path}: {e}")
            return 0, 0
        finally:
            if cap is not None:
                cap.release()

    def _extract_frames_from_video(self, video_path: Path, output_dir: Path) -> Tuple[int, int]:
        """Extract frames from a single video file using the best available method."""
        try:
            # Get video information
            video_info = self._get_video_info(video_path)
            
            # Decide which method to use
            use_ffmpeg = (
                self.use_ffmpeg and 
                self.ffmpeg_available and 
                video_info.get('is_hevc', False)
            )
            
            if use_ffmpeg:
                self.logger.debug(f"Using FFmpeg for HEVC video: {video_path.name}")
                frames_extracted, frames_skipped = self._extract_frames_with_ffmpeg(
                    video_path, output_dir, video_info
                )
                if frames_extracted > 0:
                    self.ffmpeg_processed += 1
                    return frames_extracted, frames_skipped
                else:
                    # Fallback to OpenCV
                    self.logger.warning(f"FFmpeg failed, trying OpenCV for: {video_path.name}")
            
            # Use OpenCV
            return self._extract_frames_with_opencv(video_path, output_dir, video_info)
            
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {e}")
            return 0, 0

    def _process_single_video(self, video_path: Path) -> Tuple[str, int, int]:
        """Process a single video file (wrapper for threading)."""
        try:
            # Create output directory for this video
            video_name = video_path.stem
            output_dir = self.out_path / video_name
            output_dir.mkdir(exist_ok=True)
            
            # Check if directory already contains files
            existing_files = list(output_dir.glob("*.jpg"))
            if existing_files:
                self.logger.info(f"Output directory for {video_name} already contains {len(existing_files)} files, skipping...")
                return video_name, len(existing_files), 0
            
            frames_extracted, frames_skipped = self._extract_frames_from_video(video_path, output_dir)
            return video_name, frames_extracted, frames_skipped
            
        except Exception as e:
            self.logger.error(f"Error in thread processing {video_path}: {str(e)}")
            return video_path.name, 0, 0

    def extract_all_frames(self) -> dict:
        """Extract frames from all video files using multi-threading."""
        video_files = self._get_video_files()
        if not video_files:
            self.logger.warning("No video files found in the input directory")
            return {"total_videos": 0, "processed_videos": 0, "total_frames": 0, "skipped_frames": 0}
        
        self.total_videos = len(video_files)
        self.logger.info(f"Starting extraction with {self.workers} workers")
        self.logger.info(f"FFmpeg support: {'enabled' if self.ffmpeg_available else 'disabled'}")
        self.logger.info(f"Frame validation: {'enabled' if self.validate_frames else 'disabled'}")
        
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
                        video_name, frames_extracted, frames_skipped = future.result()
                        self.processed_videos += 1
                        self.total_frames_extracted += frames_extracted
                        self.skipped_corrupted_frames += frames_skipped
                        pbar.set_postfix({
                            'Current': video_name[:20],
                            'Frames': frames_extracted,
                            'Skipped': frames_skipped
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
            "skipped_frames": self.skipped_corrupted_frames,
            "ffmpeg_processed": self.ffmpeg_processed,
            "processing_time": processing_time,
            "videos_per_second": self.processed_videos / processing_time if processing_time > 0 else 0
        }
        
        self.logger.info("=" * 50)
        self.logger.info("EXTRACTION COMPLETE")
        self.logger.info(f"Total videos found: {stats['total_videos']}")
        self.logger.info(f"Successfully processed: {stats['processed_videos']}")
        self.logger.info(f"Total frames extracted: {stats['total_frames']}")
        self.logger.info(f"Corrupted frames skipped: {stats['skipped_frames']}")
        self.logger.info(f"Videos processed with FFmpeg: {stats['ffmpeg_processed']}")
        self.logger.info(f"Processing time: {stats['processing_time']:.2f} seconds")
        self.logger.info(f"Average: {stats['videos_per_second']:.2f} videos/second")
        self.logger.info("=" * 50)
        
        return stats


def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract frames from video files with FFmpeg and OpenCV support",
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
        '--no-ffmpeg',
        action='store_true',
        help='Disable FFmpeg usage (use OpenCV only)'
    )
    
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable frame validation (save all frames including corrupted ones)'
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
            workers=args.workers,
            use_ffmpeg=not args.no_ffmpeg,
            validate_frames=not args.no_validation
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