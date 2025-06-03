import numpy as np
import time
import cProfile
import pstats
from Benchmark import create_test_card, SAMPLE_RATE, CHUNK_SIZE, N_CHANNELS
from OptimizedStream import OptimizedStream
import multiprocessing as mp
from functools import wraps
import io
import tracemalloc

# Timing decorator
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        wrapper.total_time += end - start
        wrapper.calls += 1
        return result
    wrapper.total_time = 0
    wrapper.calls = 0
    return wrapper

class ProfilingStream(OptimizedStream):
    """Stream class with detailed profiling."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sync_times = []
        self.calc_times = []
        self.memory_peaks = []
    
    @timing_decorator
    def _wait_for_completion(self, n_processes):
        """Profile the synchronization waiting time."""
        while self.sync_buffer[0] < n_processes:
            self.event_processed.wait(timeout=0.001)
            if not self.running.value:
                break
    
    def calc_next_chunk(self) -> np.ndarray:
        """Calculate next chunk with detailed timing."""
        if not self.channel_data:
            return np.array([])
        
        # Start memory tracking
        tracemalloc.start()
        
        # Reset completion counter
        sync_start = time.perf_counter()
        with self.sync_lock:
            self.sync_buffer[0] = 0
        
        # Clear processed event and set ready event
        self.event_processed.clear()
        self.event_ready.set()
        
        # Wait for processes and measure time
        n_processes = len(self.channel_batches)
        self._wait_for_completion(n_processes)
        
        sync_time = time.perf_counter() - sync_start
        self.sync_times.append(sync_time * 1000)  # Convert to ms
        
        # Record memory peak
        current, peak = tracemalloc.get_traced_memory()
        self.memory_peaks.append(peak / 1024 / 1024)  # Convert to MB
        tracemalloc.stop()
        
        # Update chunk index
        self.current_chunk.value += 1
        return self.output_buffer

def run_detailed_profile(n_chunks=50):
    """Run detailed profiling of the streaming process."""
    print("\nRunning Detailed Performance Analysis")
    print("-" * 60)
    
    # Create and run profiled stream
    card = create_test_card()
    stream = ProfilingStream(card, chunk_size=CHUNK_SIZE, channels_per_process=4)
    
    # Warmup
    for _ in range(2):
        stream.calc_next_chunk()
    
    # Profile main loop
    profiler = cProfile.Profile()
    profiler.enable()
    
    for _ in range(n_chunks):
        stream.calc_next_chunk()
    
    profiler.disable()
    
    # Calculate statistics
    sync_times = np.array(stream.sync_times)
    memory_peaks = np.array(stream.memory_peaks)
    
    # Print detailed results
    print("\nSynchronization Statistics:")
    print(f"Mean sync time: {np.mean(sync_times):.2f} ms (±{np.std(sync_times):.2f} ms)")
    print(f"Min/Max sync time: {np.min(sync_times):.2f} ms / {np.max(sync_times):.2f} ms")
    print(f"Total time in sync: {np.sum(sync_times):.2f} ms")
    
    print("\nMemory Usage Statistics:")
    print(f"Mean peak memory: {np.mean(memory_peaks):.1f} MB")
    print(f"Max peak memory: {np.max(memory_peaks):.1f} MB")
    
    print("\nWait Time Analysis:")
    print(f"Total wait calls: {stream._wait_for_completion.calls}")
    print(f"Total wait time: {stream._wait_for_completion.total_time*1000:.2f} ms")
    print(f"Average wait time per call: {stream._wait_for_completion.total_time*1000/stream._wait_for_completion.calls:.2f} ms")
    
    # Print cProfile results
    print("\nDetailed Function Profile:")
    print("-" * 60)
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())
    
    # Cleanup
    try:
        stream.__del__()
    except:
        pass

if __name__ == "__main__":
    run_detailed_profile() 