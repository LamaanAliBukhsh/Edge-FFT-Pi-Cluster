"""
Shared Memory IPC Sobel Edge Detector - Milestone 2
Implements Sobel edge detection using /dev/shm for fast inter-process communication.

This module provides:
- Shared memory array management
- Distributed Sobel computation across multiple processes
- Benchmarking utilities
- Performance comparison with threading/multiprocessing approaches
"""

import gc
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SharedMemoryManager:
    """
    Manages shared memory allocation and access using /dev/shm.
    
    Provides context manager interface for safe memory allocation and cleanup.
    """
    
    def __init__(self, shm_dir: str = "/dev/shm"):
        """
        Initialize shared memory manager.
        
        Args:
            shm_dir: Path to shared memory directory (default: /dev/shm)
        """
        self.shm_dir = shm_dir
        # Each entry is just the file path; the caller owns the memmap array reference.
        self.allocated_files = []

        if not os.path.exists(shm_dir):
            logger.warning(f"{shm_dir} not available, falling back to temp dir")
            self.shm_dir = tempfile.gettempdir()
    
    def allocate_buffer(self, name: str, dtype: np.dtype, shape: Tuple) -> Tuple[str, np.ndarray]:
        """
        Allocate a shared buffer backed by a file (in /dev/shm when available).

        Returns a (file_path, np.memmap) pair.  The caller owns the memmap and
        MUST delete their reference (and call gc.collect()) before the context
        manager exits so the file can be removed on Windows.
        """
        file_path = os.path.join(self.shm_dir, name)
        # mode='w+' creates/truncates the file and maps it read-write.
        array = np.memmap(file_path, dtype=dtype, mode='w+', shape=shape)
        self.allocated_files.append(file_path)
        logger.info(f"Allocated shared buffer: {file_path} ({shape})")
        return file_path, array
    
    def load_buffer(self, file_path: str, dtype: np.dtype, shape: Tuple) -> np.ndarray:
        """
        Open an existing shared buffer file as a read-only memmap.

        Returns a view of the on-disk data; call .copy() on the result if you
        need an independent array that outlives the file.
        """
        return np.memmap(file_path, dtype=dtype, mode='r', shape=shape)
    
    def cleanup(self):
        """Remove all allocated shared memory files.

        The caller must have already deleted any memmap references and called
        gc.collect() so that Windows releases the file handles before removal.
        """
        for file_path in self.allocated_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                logger.info(f"Cleaned up: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up {file_path}: {e}")
        self.allocated_files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class SobelEdgeDetector:
    """
    Sobel edge detection using shared memory IPC.
    
    Splits image into row blocks and distributes computation across processes.
    Uses /dev/shm for fast inter-process communication.
    """
    
    # Sobel kernels
    SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    def __init__(self, num_processes: int = 4, use_shm: bool = True):
        """
        Initialize Sobel edge detector.
        
        Args:
            num_processes: Number of worker processes
            use_shm: Whether to use shared memory (True) or regular arrays (False)
        """
        self.num_processes = num_processes
        self.use_shm = use_shm
        self.shm_manager = None
    
    @staticmethod
    def sobel_kernel(image: np.ndarray) -> np.ndarray:
        """
        Apply Sobel edge detection to an image (single-threaded reference).
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Edge-detected image
        """
        height, width = image.shape[:2]
        edges = np.zeros((height, width), dtype=np.float32)
        
        # Pad image
        padded = np.pad(image.astype(np.float32), 1, mode='edge')
        
        for i in range(height):
            for j in range(width):
                # Extract 3x3 neighborhood
                neighborhood = padded[i:i+3, j:j+3]
                
                # Apply Sobel kernels
                gx = np.sum(neighborhood * SobelEdgeDetector.SOBEL_X)
                gy = np.sum(neighborhood * SobelEdgeDetector.SOBEL_Y)
                
                # Magnitude
                edges[i, j] = np.sqrt(gx**2 + gy**2)
        
        if edges.max() > 0:
            edges = (edges / edges.max() * 255).astype(np.uint8)
        return edges.astype(np.uint8)
    
    @staticmethod
    def _worker_process(
        input_file: str,
        output_file: str,
        start_row: int,
        end_row: int,
        width: int,
        height: int,
        worker_id: int
    ):
        """
        Worker process that computes Sobel magnitudes for an assigned row block.

        Reads the full image from the shared input file so that edge-padding at
        block boundaries is always correct, then writes raw float32 magnitudes
        for the assigned rows to the shared output file.

        Vectorised with sliding_window_view + einsum — no Python pixel loops.
        """
        try:
            input_array = np.memmap(input_file, dtype=np.uint8, mode='r', shape=(height, width))
            # Output stores raw float32 magnitudes; global normalisation happens
            # in the parent after all workers finish.
            output_array = np.memmap(output_file, dtype=np.float32, mode='r+', shape=(height, width))

            # Pad full image so block-edge pixels use the correct neighbours.
            padded = np.pad(input_array.astype(np.float32), 1, mode='edge')

            # sliding_window_view gives shape (H, W, 3, 3) without any copies.
            # Slicing to [start_row:end_row] before einsum avoids full-image work.
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

            windows = sliding_window_view(padded, (3, 3))[start_row:end_row]
            gx = np.einsum('ijkl,kl->ij', windows, sobel_x)
            gy = np.einsum('ijkl,kl->ij', windows, sobel_y)
            magnitude = np.sqrt(gx ** 2 + gy ** 2)

            output_array[start_row:end_row] = magnitude
            output_array.flush()

            logger.info(f"Worker {worker_id}: Processed rows {start_row}-{end_row}")

        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
            raise
    
    def detect_edges_shm(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect edges using shared memory IPC.

        Returns:
            Tuple of (edge_detected_image uint8, computation_time seconds)
        """
        height, width = image.shape[:2]

        with SharedMemoryManager() as shm_manager:
            # Input: uint8 pixels written by this process, read by workers.
            # Output: float32 raw magnitudes written by workers, read back here.
            input_file, input_array = shm_manager.allocate_buffer(
                "sobel_input", np.uint8, (height, width)
            )
            output_file, output_array = shm_manager.allocate_buffer(
                "sobel_output", np.float32, (height, width)
            )

            # Write input to shared file and force flush to the OS page cache
            # so spawned worker processes can read it immediately.
            input_array[:] = image[:height, :width]
            input_array.flush()

            rows_per_process = height // self.num_processes

            start_time = time.time()

            processes = []
            for i in range(self.num_processes):
                start_row = i * rows_per_process
                end_row = (i + 1) * rows_per_process if i < self.num_processes - 1 else height

                p = mp.Process(
                    target=SobelEdgeDetector._worker_process,
                    args=(input_file, output_file, start_row, end_row, width, height, i)
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
                if p.exitcode != 0:
                    logger.error(f"Process {p.pid} exited with code {p.exitcode}")

            computation_time = time.time() - start_time

            # Copy result out before releasing the memmap references so that
            # cleanup() can delete the backing files (required on Windows).
            raw_magnitudes = np.array(output_array, dtype=np.float32)
            del input_array, output_array
            gc.collect()

        # Global normalisation — matches the sequential reference exactly.
        if raw_magnitudes.max() > 0:
            result = (raw_magnitudes / raw_magnitudes.max() * 255).astype(np.uint8)
        else:
            result = raw_magnitudes.astype(np.uint8)

        logger.info(f"Shared memory processing completed in {computation_time:.4f}s")
        return result, computation_time
    
    def detect_edges_sequential(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Sequential Sobel edge detection (baseline).
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (edge_detected_image, computation_time)
        """
        start_time = time.time()
        result = self.sobel_kernel(image)
        computation_time = time.time() - start_time
        
        return result, computation_time
    
    def detect_edges(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Main interface: detect edges using configured method.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (edge_detected_image, computation_time)
        """
        if self.use_shm:
            return self.detect_edges_shm(image)
        else:
            return self.detect_edges_sequential(image)


def benchmark_comparison(image: np.ndarray, num_processes: int = 4) -> Dict[str, Any]:
    """
    Benchmark shared memory vs. sequential Sobel implementation.
    
    Args:
        image: Input image as numpy array
        num_processes: Number of processes for shared memory version
        
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    # Sequential baseline
    logger.info("Running sequential baseline...")
    detector_seq = SobelEdgeDetector(num_processes=1, use_shm=False)
    edges_seq, time_seq = detector_seq.detect_edges_sequential(image)
    results['sequential'] = {
        'time': time_seq,
        'image_shape': edges_seq.shape
    }
    logger.info(f"Sequential time: {time_seq:.4f}s")
    
    # Shared memory version
    logger.info(f"Running shared memory version with {num_processes} processes...")
    detector_shm = SobelEdgeDetector(num_processes=num_processes, use_shm=True)
    edges_shm, time_shm = detector_shm.detect_edges(image)
    results['shared_memory'] = {
        'time': time_shm,
        'num_processes': num_processes,
        'image_shape': edges_shm.shape
    }
    logger.info(f"Shared memory time: {time_shm:.4f}s")
    
    # Calculate speedup
    speedup = time_seq / time_shm
    efficiency = (speedup / num_processes) * 100
    
    results['comparison'] = {
        'speedup': speedup,
        'efficiency_percent': efficiency,
        'improvement_percent': ((time_seq - time_shm) / time_seq) * 100
    }
    
    logger.info(f"Speedup: {speedup:.2f}x")
    logger.info(f"Efficiency: {efficiency:.2f}%")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Shared Memory IPC Sobel Edge Detector - Milestone 2")
    print("=" * 60)
    
    # Create a test image (grayscale gradient)
    test_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    
    print(f"\nTest image shape: {test_image.shape}")
    print(f"Test image dtype: {test_image.dtype}")
    
    # Run benchmark
    print("\n" + "-" * 60)
    print("Running Benchmarks...")
    print("-" * 60)
    
    results = benchmark_comparison(test_image, num_processes=4)
    
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"Sequential time: {results['sequential']['time']:.4f}s")
    print(f"Shared Memory time (4 procs): {results['shared_memory']['time']:.4f}s")
    print(f"Speedup: {results['comparison']['speedup']:.2f}x")
    print(f"Efficiency: {results['comparison']['efficiency_percent']:.2f}%")
    print("=" * 60)
