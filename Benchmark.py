import numpy as np
import matplotlib.pyplot as plt
from AOCard import AOCard
from Stream import Stream
import time
from typing import List, Dict, Any
import multiprocessing

# Test frequencies for each channel (Hz)
FREQUENCIES = [1e3, 2e3, 5e3, 10e3, 20e3, 50e3, 100e3, 200e3]

def create_test_card():
    """Create a test AOCard with intensive sine waveforms on all channels."""
    card = AOCard(samp_rate=10e6)  # 10 MHz sample rate
    
    # Create 8 channels, each with a different frequency sine wave
    for i, freq in enumerate(FREQUENCIES):
        ch = card.add_channel(i, default_val=0.0)
        ch.sine(0, 2.0, freq, 1.5, offset=0, phase=0)  # 2 second sine wave
        ch.compile()
    
    return card

def benchmark_chunk_timing(stream: Stream) -> List[float]:
    """Measure the time taken to compute chunks."""
    n_chunks = 20
    chunk_times = []
    
    for _ in range(n_chunks):
        start_time = time.time()
        stream.calc_next_chunk()
        chunk_times.append((time.time() - start_time) * 1000)  # Convert to ms
        
    return np.array(chunk_times)

def test_configuration(config: Dict[str, Any]) -> Dict[str, float]:
    """Test a specific configuration and return performance metrics."""
    # Create test card
    card = create_test_card()
    
    # Create stream with configuration
    start_time = time.time()
    stream = Stream(card, **config)
    init_time = time.time() - start_time
    
    # Run benchmark
    chunk_times = benchmark_chunk_timing(stream)
    
    # Calculate statistics
    mean_time = np.mean(chunk_times)
    std_time = np.std(chunk_times)
    play_time = config['chunk_size'] / 10e6 * 1000  # Convert to ms
    throughput = config['chunk_size'] * len(card.channels) / mean_time * 1000 / 1e6  # MSamples/second
    
    return {
        'mean_time': mean_time,
        'play_time': play_time,
        'throughput': throughput,
        'std_time': std_time,
        'init_time': init_time * 1000,
        'time_ratio': mean_time / play_time
    }

def run_benchmarks():
    """Run benchmarks with different configurations."""
    # Test configurations
    configs = {
        'Standard Chunks (1M)': {
            'chunk_size': int(1e6),
            'channels_per_process': 8
        },
        'Large Chunks (2M)': {
            'chunk_size': int(2e6),
            'channels_per_process': 8
        },
        'Small Chunks (500K)': {
            'chunk_size': int(5e5),
            'channels_per_process': 8
        },
        'More Processes (4 ch/proc)': {
            'chunk_size': int(1e6),
            'channels_per_process': 4
        },
        'Single Process': {
            'chunk_size': int(1e6),
            'channels_per_process': len(FREQUENCIES)
        }
    }
    
    # Run tests
    results = {}
    print("\nRunning benchmarks with different configurations:")
    print("-" * 80)
    
    for name, config in configs.items():
        print(f"\nTesting {name}...")
        results[name] = test_configuration(config)
        print(f"Mean computation: {results[name]['mean_time']:.2f} ms per chunk")
        print(f"Play time: {results[name]['play_time']:.2f} ms per chunk")
        print(f"Time ratio: {results[name]['time_ratio']:.2f}")
        print(f"Throughput: {results[name]['throughput']:.2f}M samples/second")
        print(f"Standard deviation: {results[name]['std_time']:.2f} ms")
        print(f"Initialization time: {results[name]['init_time']:.2f} ms")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    names = list(results.keys())
    mean_times = [r['mean_time'] for r in results.values()]
    play_times = [r['play_time'] for r in results.values()]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, mean_times, width, label='Computation time')
    plt.bar(x + width/2, play_times, width, label='Play time')
    
    plt.xlabel('Configuration')
    plt.ylabel('Time (ms)')
    plt.title('Computation vs Play Time for Different Configurations')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_benchmarks() 