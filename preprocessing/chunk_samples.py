# description: This script split samples into chunks based on a specified size,
# allowing for easier processing of large datasets.

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import math
import random


class DatasetSplitter:
    def __init__(self, src_dir: str, output_dir: str, chunk_size: int = 1000, shuffle: bool = False):
        """
        Initialize the dataset splitter.
        
        Args:
            src_dir: Source directory containing image and JSON files
            output_dir: Output directory for split chunks
            chunk_size: Number of image-label pairs per chunk
            shuffle: Whether to shuffle the dataset before splitting (default: False)
        """
        self.src_dir = Path(src_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        
        # Validate source directory
        if not self.src_dir.exists():
            raise ValueError(f"Source directory {src_dir} does not exist")
            
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def find_image_label_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Find all image-label pairs in the source directory.
        
        Returns:
            List of tuples containing (image_path, label_path)
        """
        # Common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        pairs = []
        
        # Find all image files and sort them for consistent ordering
        image_files = sorted([f for f in self.src_dir.iterdir() 
                            if f.suffix.lower() in image_extensions])
        
        for img_file in image_files:
            # Look for corresponding JSON file
            json_file = img_file.with_suffix('.json')
            
            if json_file.exists():
                pairs.append((img_file, json_file))
            else:
                print(f"Warning: No corresponding JSON file found for {img_file}")
        
        print(f"Found {len(pairs)} image-label pairs")
        return pairs
    
    def validate_json_file(self, json_path: Path) -> bool:
        """
        Validate that a JSON file is properly formatted.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Warning: Invalid JSON file {json_path}: {e}")
            return False
    
    def copy_pair_to_chunk(self, img_path: Path, json_path: Path, chunk_dir: Path) -> bool:
        """
        Copy an image-label pair to a chunk directory.
        
        Args:
            img_path: Source image path
            json_path: Source JSON path
            chunk_dir: Destination chunk directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Copy image file
            shutil.copy2(img_path, chunk_dir / img_path.name)
            
            # Copy JSON file
            shutil.copy2(json_path, chunk_dir / json_path.name)
            
            return True
        except Exception as e:
            print(f"Error copying {img_path.name}: {e}")
            return False
    
    def create_chunk_info(self, chunk_dir: Path, chunk_num: int, pairs_in_chunk: List[Tuple[Path, Path]]) -> None:
        """
        Create a metadata file for the chunk.
        
        Args:
            chunk_dir: Chunk directory
            chunk_num: Chunk number
            pairs_in_chunk: List of pairs in this chunk
        """
        info = {
            "chunk_number": chunk_num,
            "total_pairs": len(pairs_in_chunk),
            "files": [
                {
                    "image": pair[0].name,
                    "label": pair[1].name
                }
                for pair in pairs_in_chunk
            ]
        }
        
        info_file = chunk_dir / "chunk_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
    
    def split_dataset(self) -> Dict[str, int]:
        """
        Split the dataset into chunks.
        
        Returns:
            Dictionary with split statistics
        """
        # Find all image-label pairs
        pairs = self.find_image_label_pairs()
        
        if not pairs:
            raise ValueError("No image-label pairs found in source directory")
        
        # Validate JSON files
        valid_pairs = []
        for img_path, json_path in pairs:
            if self.validate_json_file(json_path):
                valid_pairs.append((img_path, json_path))
        
        print(f"Found {len(valid_pairs)} valid image-label pairs")
        
        # Shuffle only if explicitly requested
        if self.shuffle:
            random.shuffle(valid_pairs)
            print("Dataset shuffled")
        else:
            print("Dataset order preserved (no shuffling)")
        
        # Calculate number of chunks
        num_chunks = math.ceil(len(valid_pairs) / self.chunk_size)
        print(f"Will create {num_chunks} chunks with max {self.chunk_size} pairs each")
        
        # Split into chunks
        successful_copies = 0
        failed_copies = 0
        
        for chunk_num in range(num_chunks):
            # Create chunk directory
            chunk_dir = self.output_dir / f"chunk_{chunk_num + 1:04d}"
            chunk_dir.mkdir(exist_ok=True)
            
            # Calculate pairs for this chunk
            start_idx = chunk_num * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(valid_pairs))
            chunk_pairs = valid_pairs[start_idx:end_idx]
            
            print(f"Processing chunk {chunk_num + 1}/{num_chunks} ({len(chunk_pairs)} pairs)...")
            
            # Copy pairs to chunk directory
            chunk_successful = 0
            for img_path, json_path in chunk_pairs:
                if self.copy_pair_to_chunk(img_path, json_path, chunk_dir):
                    chunk_successful += 1
                    successful_copies += 1
                else:
                    failed_copies += 1
            
            # Create chunk info file
            self.create_chunk_info(chunk_dir, chunk_num + 1, chunk_pairs)
            
            print(f"Chunk {chunk_num + 1} completed: {chunk_successful}/{len(chunk_pairs)} pairs copied successfully")
        
        # Return statistics
        return {
            "total_pairs_found": len(pairs),
            "valid_pairs": len(valid_pairs),
            "successful_copies": successful_copies,
            "failed_copies": failed_copies,
            "chunks_created": num_chunks
        }
    
    def create_summary(self, stats: Dict[str, int]) -> None:
        """
        Create a summary file with split statistics.
        
        Args:
            stats: Statistics from the split operation
        """
        summary = {
            "split_summary": {
                "source_directory": str(self.src_dir),
                "output_directory": str(self.output_dir),
                "chunk_size": self.chunk_size,
                "shuffled": self.shuffle,
                **stats
            }
        }
        
        summary_file = self.output_dir / "split_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Split a dataset of images and JSON labels into smaller chunks")
    parser.add_argument("src_dir", help="Source directory containing image and JSON files")
    parser.add_argument("output_dir", help="Output directory for split chunks")
    parser.add_argument("--chunk-size", type=int, default=1000, 
                       help="Number of image-label pairs per chunk (default: 1000)")
    parser.add_argument("--shuffle", action="store_true", 
                       help="Shuffle the dataset before splitting (default: False)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for shuffling (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    try:
        # Initialize splitter
        splitter = DatasetSplitter(
            src_dir=args.src_dir,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            shuffle=args.shuffle
        )
        
        # Split the dataset
        print(f"Starting dataset split...")
        print(f"Source: {args.src_dir}")
        print(f"Output: {args.output_dir}")
        print(f"Chunk size: {args.chunk_size}")
        print(f"Shuffle: {args.shuffle}")
        print("-" * 50)
        
        stats = splitter.split_dataset()
        
        # Create summary
        splitter.create_summary(stats)
        
        # Print final results
        print("-" * 50)
        print("Split completed!")
        print(f"Total pairs found: {stats['total_pairs_found']}")
        print(f"Valid pairs: {stats['valid_pairs']}")
        print(f"Successfully copied: {stats['successful_copies']}")
        print(f"Failed copies: {stats['failed_copies']}")
        print(f"Chunks created: {stats['chunks_created']}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())