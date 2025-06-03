import numpy as np
import time
from Benchmark import create_test_card, SAMPLE_RATE, CHUNK_SIZE, N_CHANNELS
from Stream import Stream
from OptimizedStream import OptimizedStream
from HighPerformanceStream import HighPerformanceStream
import matplotlib.pyplot as plt

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

def plot_comparison(results_dict):
    """Plot comparison of different implementations."""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Processing time distributions
    plt.subplot(1, 3, 1)
    data = [results['chunk_times'] for results in results_dict.values()]
    labels = list(results_dict.keys())
    parts = plt.violinplot(data, range(len(data)), points=20, widths=0.7)
    
    # Customize violin plot
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    
    # Add play time threshold
    play_time = results_dict[labels[0]]['play_time']
    plt.axhline(y=play_time, color='r', linestyle='--', label=f'Real-time threshold ({play_time:.1f} ms)')
    
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.ylabel('Processing Time (ms)')
    plt.title('Chunk Processing Time Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Throughput comparison
    plt.subplot(1, 3, 2)
    throughputs = [results['throughput'] for results in results_dict.values()]
    bars = plt.bar(range(len(throughputs)), throughputs, width=0.7)
    
    # Add value labels
    for i, v in enumerate(throughputs):
        plt.text(i, v, f'{v:.1f}M', ha='center', va='bottom')
    
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.ylabel('Throughput (M samples/second)')
    plt.title('Throughput Comparison')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Timing violations
    plt.subplot(1, 3, 3)
    violations = []
    for results in results_dict.values():
        n_violations = np.sum(results['chunk_times'] > results['play_time'])
        violations.append((n_violations / len(results['chunk_times'])) * 100)
    
    bars = plt.bar(range(len(violations)), violations, width=0.7)
    
    # Add value labels
    for i, v in enumerate(violations):
        plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.ylabel('Timing Violations (%)')
    plt.title('Real-time Violations')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    print("\nRunning Stream Implementation Comparison")
    print(f"Settings:")
    print(f"- Sample Rate: {SAMPLE_RATE/1e6:.1f} MHz")
    print(f"- Channels: {N_CHANNELS}")
    print(f"- Chunk Size: {CHUNK_SIZE:,} samples")
    print("-" * 50)
    
    # Test all implementations
    implementations = {
        'Original': Stream,
        'Optimized': OptimizedStream,
        'High-Performance': HighPerformanceStream
    }
    
    results = {}
    for name, impl in implementations.items():
        print(f"\nTesting {name} implementation...")
        results[name] = benchmark_stream(impl)
        print_results(name, results[name])
    
    # Print overall comparison
    print("\nPerformance Comparison:")
    print("-" * 50)
    baseline_time = results['Original']['mean_time']
    baseline_throughput = results['Original']['throughput']
    
    for name in ['Optimized', 'High-Performance']:
        time_improvement = (baseline_time - results[name]['mean_time']) / baseline_time * 100
        throughput_increase = (results[name]['throughput'] - baseline_throughput) / baseline_throughput * 100
        print(f"\n{name} vs Original:")
        print(f"Processing time improvement: {time_improvement:.1f}%")
        print(f"Throughput increase: {throughput_increase:.1f}%")
    
    # Plot comparison
    plot_comparison(results)

if __name__ == "__main__":
    main() 