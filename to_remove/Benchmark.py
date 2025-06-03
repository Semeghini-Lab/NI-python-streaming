import numpy as np
import matplotlib.pyplot as plt
from AOCard import AOCard
from Stream import Stream
import time
import signal
import sys
import gc
import multiprocessing as mp
import psutil
import os

# Test parameters
SAMPLE_RATE = 10e6  # 10 MHz
CLIP_DURATION = 0.5  # seconds
CHUNK_SIZE = 500000  # 500K samples
N_CHANNELS = 16
N_WARMUP = 2

def set_high_priority():
    """Set process to high priority."""
    try:
        p = psutil.Process(os.getpid())
        if sys.platform == 'win32':
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            p.nice(-10)  # Lower nice value = higher priority
    except:
        pass

def kill_child_processes():
    """Aggressively kill all child processes."""
    for p in mp.active_children():
        try:
            p.terminate()
            p.join(timeout=0.1)
            if p.is_alive():
                p.kill()
        except:
            pass
    gc.collect()

def create_test_card():
    """Create a test card with sine waves of increasing frequencies."""
    card = AOCard(samp_rate=SAMPLE_RATE)
    frequencies = np.logspace(np.log10(1e3), np.log10(200e3), N_CHANNELS)
    for i, freq in enumerate(frequencies):
        ch = card.add_channel(i, default_val=0.0)
        ch.sine(0, CLIP_DURATION, freq, 1.5, offset=0, phase=0)
        ch.compile()
    return card

def run_single_test(ch_per_proc):
    """Run a single benchmark test."""
    stream = None
    try:
        # Setup
        card = create_test_card()
        stream = Stream(card, chunk_size=CHUNK_SIZE, channels_per_process=ch_per_proc)
        total_samples = int(CLIP_DURATION * SAMPLE_RATE)
        n_chunks = (total_samples + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        # Pre-allocate arrays for better performance
        chunk_times = np.empty(n_chunks)
        
        # Warmup phase with high priority
        set_high_priority()
        for _ in range(N_WARMUP):
            stream.calc_next_chunk()
        
        # Measure with high precision timer
        start_total = time.perf_counter()
        
        for i in range(n_chunks):
            start = time.perf_counter()
            stream.calc_next_chunk()
            chunk_times[i] = (time.perf_counter() - start) * 1000
        
        total_time = time.perf_counter() - start_total
        
        # Calculate metrics
        mean_time = np.mean(chunk_times)
        std_time = np.std(chunk_times)
        play_time = CHUNK_SIZE / SAMPLE_RATE * 1000
        throughput = CHUNK_SIZE * N_CHANNELS / mean_time * 1000 / 1e6
        
        return {
            'mean_time': mean_time,
            'std_time': std_time,
            'play_time': play_time,
            'throughput': throughput,
            'chunk_times': chunk_times,
            'total_time': total_time
        }
    
    except Exception as e:
        print(f"Error in test: {e}")
        return None
    
    finally:
        if stream is not None:
            try:
                stream.__del__()
            except:
                pass
        kill_child_processes()
        time.sleep(0.1)  # Reduced cooldown

def print_results(ch_per_proc, results):
    """Print benchmark results."""
    if results is None:
        print(f"\n{ch_per_proc} channels/process: Failed")
        return
        
    print(f"\n{ch_per_proc} channels/process ({N_CHANNELS//ch_per_proc} processes)")
    print(f"- Mean compute time: {results['mean_time']:.2f} ms (±{results['std_time']:.2f} ms)")
    print(f"- Play time: {results['play_time']:.2f} ms")
    print(f"- Time ratio: {results['mean_time']/results['play_time']:.2f}")
    print(f"- Throughput: {results['throughput']:.2f}M samples/second")
    print(f"- Total time: {results['total_time']:.2f}s")

def plot_results(results_4ch, results_8ch):
    """Plot benchmark results."""
    if results_4ch is None and results_8ch is None:
        return
        
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Processing time distribution
    plt.subplot(1, 2, 1)
    positions = [0, 2]
    data = []
    labels = []
    play_times = []
    
    if results_4ch is not None:
        data.append(results_4ch['chunk_times'])
        labels.append("4 ch/proc")
        play_times.append(results_4ch['play_time'])
    if results_8ch is not None:
        data.append(results_8ch['chunk_times'])
        labels.append("2 ch/proc")
        play_times.append(results_8ch['play_time'])
    
    if data:
        parts = plt.violinplot(data, positions[:len(data)], points=20, widths=1.5)
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        # Plot real-time threshold
        play_time = play_times[0]  # Should be the same for all configurations
        plt.axhline(y=play_time, color='r', linestyle='--', label=f'Real-time threshold ({play_time:.1f} ms)')
        
        # Add timing violation statistics
        for i, times in enumerate(data):
            violations = np.sum(times > play_time)
            violation_pct = (violations / len(times)) * 100
            max_time = np.max(times)
            plt.text(positions[i], plt.ylim()[1]*0.95, 
                    f'Violations: {violations}/{len(times)}\n({violation_pct:.1f}%)\nMax: {max_time:.1f} ms',
                    horizontalalignment='center', verticalalignment='top')
        
        plt.xticks(positions[:len(data)], labels)
        plt.ylabel('Processing Time (ms)')
        plt.title('Chunk Processing Time Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Plot 2: Throughput comparison
    plt.subplot(1, 2, 2)
    throughputs = []
    if results_4ch is not None:
        throughputs.append(results_4ch['throughput'])
    if results_8ch is not None:
        throughputs.append(results_8ch['throughput'])
    
    if throughputs:
        bars = plt.bar(positions[:len(throughputs)], throughputs, width=1.5)
        
        # Add value labels on top of bars
        for i, v in enumerate(throughputs):
            plt.text(positions[i], v, f'{v:.1f}M', 
                    horizontalalignment='center', verticalalignment='bottom')
        
        plt.xticks(positions[:len(throughputs)], labels)
        plt.ylabel('Throughput (M samples/second)')
        plt.title('Throughput Comparison')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def run_benchmark():
    """Run benchmark with aggressive cleanup."""
    # Set up signal handler for clean exit
    signal.signal(signal.SIGINT, lambda s, f: (kill_child_processes(), sys.exit(0)))
    
    print("\nRunning Quick Benchmark")
    print(f"Settings:")
    print(f"- Sample Rate: {SAMPLE_RATE/1e6:.1f} MHz")
    print(f"- Channels: {N_CHANNELS}")
    print(f"- Chunk Size: {CHUNK_SIZE:,} samples")
    print(f"- Duration: {CLIP_DURATION:.1f} seconds")
    print("-" * 50)
    
    # Initial cleanup
    kill_child_processes()
    
    # Run tests with high priority
    set_high_priority()
    
    # Run tests
    results_4ch = run_single_test(4)  # 4 channels per process
    print_results(4, results_4ch)
    
    results_8ch = run_single_test(2)  # 2 channels per process (8 processes)
    print_results(2, results_8ch)
    
    # Plot if we have results
    plot_results(results_4ch, results_8ch)
    
    # Final cleanup
    kill_child_processes()

if __name__ == "__main__":
    run_benchmark() 