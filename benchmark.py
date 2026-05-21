#!/usr/bin/env python3
"""
Benchmark script for gesture recognition model performance.
"""

import time
import argparse
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path


class PerformanceBenchmark:
    """Benchmark different components of the gesture recognition system."""
    
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load model for benchmarking."""
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded: {self.model.name}")
    
    def benchmark_inference(self, iterations=100, batch_size=1):
        """Benchmark inference speed."""
        if not self.model:
            print("Error: Model not loaded")
            return
        
        # Create dummy input
        input_shape = self.model.input_shape
        dummy_input = np.random.rand(batch_size, *input_shape[1:])
        
        # Warmup
        for _ in range(10):
            _ = self.model.predict(dummy_input, verbose=0)
        
        # Benchmark
        print(f"\nBenchmarking inference ({iterations} iterations, batch_size={batch_size})...")
        times = []
        
        for i in range(iterations):
            start = time.perf_counter()
            _ = self.model.predict(dummy_input, verbose=0)
            end = time.perf_counter()
            times.append((end - start) * 1000)
            
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i+1}/{iterations}: {times[-1]:.2f}ms")
        
        times = np.array(times)
        fps = 1000 / times.mean()
        
        print(f"\nInference Results:")
        print(f"  Mean: {times.mean():.2f}ms")
        print(f"  Std: {times.std():.2f}ms")
        print(f"  Min: {times.min():.2f}ms")
        print(f"  Max: {times.max():.2f}ms")
        print(f"  FPS: {fps:.1f}")
        
        return times
    
    def benchmark_preprocessing(self, iterations=1000):
        """Benchmark image preprocessing pipeline."""
        # Create dummy frame
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        roi = frame[200:400, 400:600]
        
        print(f"\nBenchmarking preprocessing ({iterations} iterations)...")
        
        # Binary mask preprocessing
        times_binary = []
        for i in range(iterations):
            start = time.perf_counter()
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
            _, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            end = time.perf_counter()
            times_binary.append((end - start) * 1000)
            
            if (i + 1) % 100 == 0:
                print(f"  Iteration {i+1}/{iterations}: {times_binary[-1]:.2f}ms")
        
        times_binary = np.array(times_binary)
        
        print(f"\nBinary Mask Preprocessing Results:")
        print(f"  Mean: {times_binary.mean():.2f}ms")
        print(f"  Std: {times_binary.std():.2f}ms")
        print(f"  Min: {times_binary.min():.2f}ms")
        print(f"  Max: {times_binary.max():.2f}ms")
        
        return times_binary
    
    def benchmark_model_size(self):
        """Benchmark model size and parameters."""
        if not self.model:
            print("Error: Model not loaded")
            return
        
        print(f"\nModel Information:")
        print(f"  Name: {self.model.name}")
        print(f"  Total layers: {len(self.model.layers)}")
        
        total_params = self.model.count_params()
        print(f"  Total parameters: {total_params:,}")
        
        # Estimate model size
        model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        print(f"  Estimated size (float32): {model_size_mb:.2f} MB")
        print(f"  Estimated size (float16): {model_size_mb/2:.2f} MB")
        print(f"  Estimated size (int8): {model_size_mb/4:.2f} MB")
        
        # Layer breakdown
        print(f"\nLayer-wise breakdown:")
        for layer in self.model.layers:
            params = layer.count_params()
            if params > 0:
                print(f"  {layer.name}: {params:,} parameters")
    
    def benchmark_gpu_availability(self):
        """Check GPU availability and settings."""
        print(f"\nGPU Availability Check:")
        
        gpus = tf.config.list_physical_devices('GPU')
        cpus = tf.config.list_physical_devices('CPU')
        
        print(f"  Physical GPUs: {len(gpus)}")
        print(f"  Physical CPUs: {len(cpus)}")
        
        if gpus:
            for gpu in gpus:
                print(f"    GPU: {gpu}")
            print(f"\n  GPU memory growth enabled: {tf.config.experimental.get_memory_growth(gpus[0])}")
        else:
            print("  No GPU detected - using CPU only")


def main():
    parser = argparse.ArgumentParser(description="Benchmark gesture recognition model")
    parser.add_argument('--mode', choices=['inference', 'preprocessing', 'model', 'gpu', 'all'],
                       default='all', help='Benchmark mode')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark(args.model)
    
    if args.mode in ['inference', 'all']:
        benchmark.benchmark_inference(args.iterations, args.batch_size)
    
    if args.mode in ['preprocessing', 'all']:
        benchmark.benchmark_preprocessing(args.iterations)
    
    if args.mode in ['model', 'all']:
        benchmark.benchmark_model_size()
    
    if args.mode in ['gpu', 'all']:
        benchmark.benchmark_gpu_availability()


if __name__ == '__main__':
    main()
