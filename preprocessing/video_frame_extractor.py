import argparse
import os
import sys
import cv2
import logging
from time import time
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import numpy as np
import subprocess
import tempfile
import json
from datetime import datetime


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
        
        # Setup logging (will be configured in main)
        self.logger = logging.getLogger(__name__)
        
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

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            # Use check=False to prevent raising an exception on non-zero exit codes
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5, check=False)
            if result.returncode == 0:
                self.logger.info("FFmpeg detected and available")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
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
        video_files = [
            p for p in self.root_path.rglob('*') if p.is_file() and p.suffix.lower() in self.VIDEO_EXTENSIONS
        ] if recursive else [
            p for p in self.root_path.iterdir() if p.is_file() and p.suffix.lower() in self.VIDEO_EXTENSIONS
        ]
        
        self.logger.info(f"Found {len(video_files)} video files{'(recursive)' if recursive else ''}")
        return video_files

    def _generate_video_timestamp(self, video_path: Path) -> str:
        """
        Generate a unique timestamp for the video based on current time and video name.
        """
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        video_name_hash = abs(hash(video_path.name)) % 10000
        return f"{timestamp}_{video_name_hash:04d}"

    def _get_video_info(self, video_path: Path) -> dict:
        """Get video information using FFprobe, with a fallback to OpenCV."""
        info = {'codec': 'unknown', 'is_hevc': False, 'fps': 25.0}
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'stream=codec_name,width,height,duration,r_frame_rate,codec_type',
                '-of', 'json', str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        info['codec'] = stream.get('codec_name', 'unknown').lower()
                        info['is_hevc'] = info['codec'] in self.HEVC_CODECS
                        info['width'] = stream.get('width')
                        info['height'] = stream.get('height')
                        if 'duration' in stream:
                            try: info['duration'] = float(stream['duration'])
                            except (ValueError, TypeError): pass
                        if 'r_frame_rate' in stream and '/' in stream['r_frame_rate']:
                            try:
                                num, den = stream['r_frame_rate'].split('/')
                                if float(den) > 0: info['fps'] = float(num) / float(den)
                            except (ValueError, ZeroDivisionError): pass
                        break
                return info
        except (Exception, json.JSONDecodeError) as e:
            self.logger.debug(f"FFprobe failed for {video_path.name}: {e}")

        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                info['fps'] = cap.get(cv2.CAP_PROP_FPS) or 25.0
                info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if info['fps'] > 0: info['duration'] = frame_count / info['fps']
                cap.release()
        except Exception as e:
            self.logger.debug(f"OpenCV info fallback failed for {video_path.name}: {e}")
        
        return info

    def _is_frame_valid(self, frame: np.ndarray) -> bool:
        """Check if a frame is valid (not corrupted/gray)."""
        if frame is None or frame.size == 0: return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        if np.var(gray) < 50.0: return False
        mean_val = np.mean(gray)
        if mean_val < 5 or mean_val > 250: return False
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        if np.max(hist) > 0.9 * gray.size: return False
        return True

    def _extract_frames_with_method(self, video_path: Path, output_dir: Path, video_info: dict, timestamp_prefix: str, use_ffmpeg: bool) -> Tuple[int, int]:
        """Core frame extraction logic for either FFmpeg or OpenCV."""
        saved_count, skipped_count = 0, 0
        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
        
        if use_ffmpeg:
            try:
                fps = video_info.get('fps', 25.0) or 25.0
                extract_fps = fps / self.skip_frame
                with tempfile.TemporaryDirectory() as temp_dir:
                    cmd = ['ffmpeg', '-y', '-i', str(video_path), '-vf', f'fps={extract_fps}', '-q:v', '2', '-f', 'image2', str(Path(temp_dir) / 'f_%08d.jpg')]
                    subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
                    for frame_file in sorted(Path(temp_dir).glob('f_*.jpg')):
                        frame = cv2.imread(str(frame_file))
                        if self.validate_frames and not self._is_frame_valid(frame):
                            skipped_count += 1
                            continue
                        filename = f"{timestamp_prefix}_{saved_count:06d}.jpg"
                        if cv2.imwrite(str(output_dir / filename), frame, jpeg_params): saved_count += 1
                        else: skipped_count += 1
                return saved_count, skipped_count
            except Exception as e:
                self.logger.error(f"FFmpeg extraction error for {video_path.name}: {e}")
                return 0, 0
        else: # OpenCV
            cap = None
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    self.logger.error(f"Failed to open with OpenCV: {video_path.name}")
                    return 0, 0
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    if frame_idx % self.skip_frame == 0:
                        if self.validate_frames and not self._is_frame_valid(frame):
                            skipped_count += 1
                        else:
                            filename = f"{timestamp_prefix}_{saved_count:06d}.jpg"
                            if cv2.imwrite(str(output_dir / filename), frame, jpeg_params): saved_count += 1
                            else: skipped_count += 1
                    frame_idx += 1
                return saved_count, skipped_count
            except Exception as e:
                self.logger.error(f"OpenCV extraction error for {video_path.name}: {e}")
                return 0, 0
            finally:
                if cap: cap.release()

    def _process_single_video(self, video_path: Path) -> Tuple[str, int, int, bool]:
        """Wrapper for processing a single video."""
        output_dir = self.out_path / video_path.stem
        output_dir.mkdir(exist_ok=True)
        
        if any(output_dir.glob('*.jpg')):
            self.logger.info(f"Skipping non-empty directory: {video_path.stem}")
            return video_path.stem, len(list(output_dir.glob('*.jpg'))), 0, False

        timestamp_prefix = self._generate_video_timestamp(video_path)
        video_info = self._get_video_info(video_path)
        
        use_ffmpeg_flag = self.use_ffmpeg and self.ffmpeg_available and video_info.get('is_hevc', False)
        
        if use_ffmpeg_flag:
            self.logger.debug(f"Using FFmpeg for HEVC video: {video_path.name}")
            frames_extracted, frames_skipped = self._extract_frames_with_method(video_path, output_dir, video_info, timestamp_prefix, use_ffmpeg=True)
            if frames_extracted > 0 or frames_skipped > 0:
                return video_path.stem, frames_extracted, frames_skipped, True
            self.logger.warning(f"FFmpeg failed for {video_path.name}, falling back to OpenCV.")
        
        self.logger.debug(f"Using OpenCV for video: {video_path.name}")
        frames_extracted, frames_skipped = self._extract_frames_with_method(video_path, output_dir, video_info, timestamp_prefix, use_ffmpeg=False)
        return video_path.stem, frames_extracted, frames_skipped, False

    def extract_all_frames(self) -> dict:
        """Extract frames from all videos using a thread pool."""
        video_files = self._get_video_files()
        if not video_files:
            self.logger.warning("No video files found.")
            return {}
        
        self.total_videos = len(video_files)
        self.logger.info(f"Starting extraction with {self.workers} workers...")
        
        start_time = time()
        
        with logging_redirect_tqdm(), ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_video = {executor.submit(self._process_single_video, v): v for v in video_files}
            
            with tqdm(total=len(video_files), desc="Processing Videos", unit="video") as pbar:
                for future in as_completed(future_to_video):
                    try:
                        video_name, extracted, skipped, used_ffmpeg = future.result()
                        if extracted > 0 or skipped > 0: self.processed_videos += 1
                        if used_ffmpeg: self.ffmpeg_processed += 1
                        self.total_frames_extracted += extracted
                        self.skipped_corrupted_frames += skipped
                        pbar.set_postfix({'Current': video_name[:20], 'Frames': extracted, 'Skipped': skipped})
                    except Exception as e:
                        self.logger.error(f"A video process failed: {e}")
                    finally:
                        pbar.update(1)

        processing_time = time() - start_time
        stats = {
            "total_videos": self.total_videos, "processed_videos": self.processed_videos,
            "total_frames": self.total_frames_extracted, "skipped_frames": self.skipped_corrupted_frames,
            "ffmpeg_processed": self.ffmpeg_processed, "processing_time": processing_time,
            "videos_per_second": self.processed_videos / processing_time if processing_time > 0 else 0
        }
        
        self.logger.info("\n" + "="*50 + "\nEXTRACTION COMPLETE\n" +
                         f"Total videos found: {stats['total_videos']}\n" +
                         f"Successfully processed: {stats['processed_videos']}\n" +
                         f"Total frames extracted: {stats['total_frames']}\n" +
                         f"Corrupted frames skipped: {stats['skipped_frames']}\n" +
                         f"Videos processed with FFmpeg: {stats['ffmpeg_processed']}\n" +
                         f"Processing time: {stats['processing_time']:.2f} seconds\n" +
                         f"Average: {stats['videos_per_second']:.2f} videos/second\n" + "="*50)
        return stats

def setup_logging(level: str):
    """Configures the root logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('video_extraction.log'), logging.StreamHandler()]
    )

def main():
    """Main entry point with robust terminal state handling."""
    exit_code = 0
    try:
        args = parse_arguments()
        setup_logging(args.log_level)
        
        extractor = VideoFrameExtractor(
            root_path=args.root_path, out_path=args.out_path, skip_frame=args.skip,
            jpg_quality=args.jpg_quality, workers=args.workers,
            use_ffmpeg=not args.no_ffmpeg, validate_frames=not args.no_validation
        )
        
        stats = extractor.extract_all_frames()
        
        if not stats: exit_code = 1
        elif stats.get('processed_videos', 0) < stats.get('total_videos', 0): exit_code = 2

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        exit_code = 130
    except Exception as e:
        logging.getLogger(__name__).error(f"Fatal error: {str(e)}", exc_info=True)
        exit_code = 1
    finally:
        # **CRITICAL:** This block ensures the terminal is restored.
        # It runs on both normal exit and exceptions.
        if sys.platform != "win32":
            # This command restores terminal settings on Linux/macOS.
            os.system('stty echo')
        sys.exit(exit_code)

def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract frames from video files with consistent naming per video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('root_path', help='Input directory containing video files')
    parser.add_argument('out_path', help='Output directory for extracted frames')
    parser.add_argument('--skip', type=int, default=5, help='Number of frames to skip')
    parser.add_argument('--jpg_quality', type=int, default=80, help='JPEG quality (1-100)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker threads')
    parser.add_argument('--no-ffmpeg', action='store_true', help='Disable FFmpeg usage')
    parser.add_argument('--no-validation', action='store_true', help='Disable frame validation')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Logging level')
    return parser.parse_args()

if __name__ == '__main__':
    main()