import numpy as np
import time
from Benchmark import create_test_card, SAMPLE_RATE, CHUNK_SIZE, N_CHANNELS
from Stream import Stream
from OptimizedStream import OptimizedStream

def benchmark_stream(stream_class, n_chunks=50, warmup=2):
    """Benchmark a stream class implementation."""
    # Setup
    card = create_test_card()
    stream = stream_class(card, chunk_size=CHUNK_SIZE, channels_per_process=4)
    
    # Warmup
    for _ in range(warmup):
        stream.calc_next_chunk()
    
    # Measure chunk times
    chunk_times = []
    start_total = time.perf_counter()
    
    for _ in range(n_chunks):
        start = time.perf_counter()
        stream.calc_next_chunk()
        chunk_times.append((time.perf_counter() - start) * 1000)  # Convert to ms
    
    total_time = time.perf_counter() - start_total
    
    # Calculate metrics
    chunk_times = np.array(chunk_times)
    mean_time = np.mean(chunk_times)
    std_time = np.std(chunk_times)
    min_time = np.min(chunk_times)
    max_time = np.max(chunk_times)
    play_time = CHUNK_SIZE / SAMPLE_RATE * 1000  # Convert to ms
    throughput = CHUNK_SIZE * N_CHANNELS / mean_time * 1000 / 1e6  # M samples/second
    
    # Cleanup
    try:
        stream.__del__()
    except:
        pass
    
    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'play_time': play_time,
        'throughput': throughput,
        'total_time': total_time,
        'chunk_times': chunk_times
    }

def print_results(name, results):
    """Print benchmark results in a formatted way."""
    print(f"\n{name} Results:")
    print("-" * 50)
    print(f"Mean compute time: {results['mean_time']:.2f} ms (±{results['std_time']:.2f} ms)")
    print(f"Min/Max time: {results['min_time']:.2f} ms / {results['max_time']:.2f} ms")
    print(f"Play time: {results['play_time']:.2f} ms")
    print(f"Time ratio: {results['mean_time']/results['play_time']:.2f}")
    print(f"Throughput: {results['throughput']:.2f}M samples/second")
    print(f"Total time: {results['total_time']:.2f}s")
    
    # Calculate timing violations
    violations = np.sum(results['chunk_times'] > results['play_time'])
    violation_pct = (violations / len(results['chunk_times'])) * 100
    print(f"Timing violations: {violations}/{len(results['chunk_times'])} ({violation_pct:.1f}%)")

def main():
    print("\nRunning Stream Comparison Benchmark")
    print(f"Settings:")
    print(f"- Sample Rate: {SAMPLE_RATE/1e6:.1f} MHz")
    print(f"- Channels: {N_CHANNELS}")
    print(f"- Chunk Size: {CHUNK_SIZE:,} samples")
    print("-" * 50)
    
    # Test original Stream
    print("\nTesting original Stream implementation...")
    original_results = benchmark_stream(Stream)
    print_results("Original Stream", original_results)
    
    # Test optimized Stream
    print("\nTesting optimized Stream implementation...")
    optimized_results = benchmark_stream(OptimizedStream)
    print_results("Optimized Stream", optimized_results)
    
    # Print improvement metrics
    improvement = (original_results['mean_time'] - optimized_results['mean_time']) / original_results['mean_time'] * 100
    throughput_increase = (optimized_results['throughput'] - original_results['throughput']) / original_results['throughput'] * 100
    
    print("\nPerformance Comparison:")
    print("-" * 50)
    print(f"Processing time improvement: {improvement:.1f}%")
    print(f"Throughput increase: {throughput_increase:.1f}%")

if __name__ == "__main__":
    main() 