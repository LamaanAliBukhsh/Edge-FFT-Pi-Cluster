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
        Pure-Python double-loop Sobel reference (NAIVE baseline).

        Kept for educational comparison only -- this kernel is roughly two
        orders of magnitude slower than the vectorised version because it
        runs the inner 3x3 multiply-accumulate from Python.  Quoting the
        speedup of the parallel pipeline against THIS baseline conflates
        algorithmic gains (NumPy vectorisation) with parallelism gains
        and produces physically impossible efficiency numbers (>100%).

        For honest parallel speedup measurements, see
        ``sobel_kernel_vectorized`` and ``detect_edges_vectorized``.
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
    def sobel_kernel_vectorized(image: np.ndarray) -> np.ndarray:
        """
        Vectorised single-process Sobel (FAIR parallel-speedup baseline).

        Uses exactly the same einsum-based kernel as the per-worker
        compute path in ``_worker_process``; the only difference is that
        the entire image is processed in this process instead of being
        split across N workers.  This is therefore the correct denominator
        for the parallel speedup S(N) = T_vec_serial / T_parallel(N).
        """
        padded = np.pad(image.astype(np.float32), 1, mode='edge')
        windows = sliding_window_view(padded, (3, 3))
        gx = np.einsum('ijkl,kl->ij', windows, SobelEdgeDetector.SOBEL_X)
        gy = np.einsum('ijkl,kl->ij', windows, SobelEdgeDetector.SOBEL_Y)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        if magnitude.max() > 0:
            magnitude = (magnitude / magnitude.max() * 255)
        return magnitude.astype(np.uint8)
    
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
        Naive (pure-Python) sequential Sobel.  See ``sobel_kernel`` for the
        warning -- this is the SLOW baseline used only to demonstrate the
        cost of running the inner loop in Python.
        """
        start_time = time.time()
        result = self.sobel_kernel(image)
        computation_time = time.time() - start_time

        return result, computation_time

    def detect_edges_vectorized(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Vectorised single-process Sobel.  This is the correct baseline for
        computing parallel speedup S(N) of the shared-memory pipeline.
        """
        start_time = time.time()
        result = self.sobel_kernel_vectorized(image)
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
    Three-way benchmark of the Sobel pipeline:

        1. Naive (pure-Python) single process    -- educational baseline
        2. Vectorised (einsum) single process    -- FAIR baseline for S(N)
        3. Shared-memory parallel (N processes)  -- the M2 deliverable

    Reports two speedup numbers so the apples-to-apples parallel speedup
    cannot be confused with the (much larger) algorithmic vectorisation
    win.  Efficiency is calculated against the vectorised baseline only,
    which is why it stays bounded by 100% as physics requires.
    """
    results: Dict[str, Any] = {}

    # 1. Naive (pure-Python) sequential baseline
    logger.info("Running NAIVE pure-Python sequential baseline...")
    detector_naive = SobelEdgeDetector(num_processes=1, use_shm=False)
    edges_naive, time_naive = detector_naive.detect_edges_sequential(image)
    results['naive_sequential'] = {
        'time': time_naive,
        'image_shape': edges_naive.shape,
    }
    logger.info(f"Naive sequential time: {time_naive:.4f}s")

    # 2. Vectorised single-process baseline -- the correct denominator for S(N)
    logger.info("Running VECTORISED single-process baseline (fair S(N) baseline)...")
    detector_vec = SobelEdgeDetector(num_processes=1, use_shm=False)
    edges_vec, time_vec = detector_vec.detect_edges_vectorized(image)
    results['vectorized_sequential'] = {
        'time': time_vec,
        'image_shape': edges_vec.shape,
    }
    logger.info(f"Vectorised sequential time: {time_vec:.4f}s")

    # 3. Shared-memory parallel pipeline
    logger.info(f"Running SHARED-MEMORY parallel pipeline with {num_processes} processes...")
    detector_shm = SobelEdgeDetector(num_processes=num_processes, use_shm=True)
    edges_shm, time_shm = detector_shm.detect_edges(image)
    results['shared_memory'] = {
        'time': time_shm,
        'num_processes': num_processes,
        'image_shape': edges_shm.shape,
    }
    logger.info(f"Shared memory time: {time_shm:.4f}s")

    # Two distinct speedup ratios.  Only the second one is a parallelism
    # measurement; the first is mostly the cost of running NumPy code from
    # Python instead of from C.
    vectorisation_speedup = time_naive / time_vec if time_vec > 0 else float('inf')
    parallel_speedup = time_vec / time_shm if time_shm > 0 else float('inf')
    parallel_efficiency = (parallel_speedup / num_processes) * 100
    end_to_end_speedup = time_naive / time_shm if time_shm > 0 else float('inf')

    results['comparison'] = {
        'vectorisation_speedup': vectorisation_speedup,
        'parallel_speedup': parallel_speedup,
        'parallel_efficiency_percent': parallel_efficiency,
        'end_to_end_speedup': end_to_end_speedup,
    }

    logger.info(
        f"Vectorisation gain (algorithm only): {vectorisation_speedup:.2f}x  "
        f"-- not parallelism, do not quote as such"
    )
    logger.info(
        f"Parallel speedup S({num_processes}) = T_vec / T_par : "
        f"{parallel_speedup:.2f}x  (efficiency {parallel_efficiency:.1f}%)"
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Shared-memory IPC Sobel benchmark (Milestone 2)"
    )
    parser.add_argument("--size", type=int, default=256,
                        help="Side length of the synthetic square test image (default 256)")
    parser.add_argument("--processes", type=int, default=4,
                        help="Number of parallel worker processes (default 4)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the synthetic image (default 0)")
    args = parser.parse_args()

    print("=" * 60)
    print("Shared Memory IPC Sobel Edge Detector - Milestone 2")
    print("=" * 60)

    rng = np.random.default_rng(args.seed)
    test_image = rng.integers(0, 256, (args.size, args.size), dtype=np.uint8)

    print(f"\nTest image shape: {test_image.shape}")
    print(f"Test image dtype: {test_image.dtype}")
    print(f"Worker processes: {args.processes}")

    print("\n" + "-" * 60)
    print("Running Benchmarks...")
    print("-" * 60)

    results = benchmark_comparison(test_image, num_processes=args.processes)

    cmp = results['comparison']
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"  Naive  sequential   (pure-Python loop) : "
          f"{results['naive_sequential']['time']:.4f}s")
    print(f"  Vector sequential   (einsum, 1 proc)   : "
          f"{results['vectorized_sequential']['time']:.4f}s")
    print(f"  Shared-memory paral ({args.processes} procs)         : "
          f"{results['shared_memory']['time']:.4f}s")
    print("-" * 60)
    print(f"  Vectorisation gain   (algorithm only)  : "
          f"{cmp['vectorisation_speedup']:.2f}x  [NOT parallelism]")
    print(f"  Parallel speedup S({args.processes})                : "
          f"{cmp['parallel_speedup']:.2f}x  (efficiency "
          f"{cmp['parallel_efficiency_percent']:.1f}%)")
    print(f"  End-to-end speedup   (vec x parallel)  : "
          f"{cmp['end_to_end_speedup']:.2f}x")
    print("=" * 60)
    print("Quote 'Parallel speedup S(N)' for the M2 / M3 reports.")
    print("It is bounded by N (here {n}) as required by Amdahl's law.".format(n=args.processes))
    print("=" * 60)
