import cProfile
import pstats
from pstats import SortKey
import Benchmark

def profile_benchmark():
    # Run with profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run benchmark
    Benchmark.run_benchmark(chunk_size=1e6, channels_per_process=2, plot_results=False)
    
    profiler.disable()
    
    # Print sorted statistics
    stats = pstats.Stats(profiler)
    print("\nProfile sorted by cumulative time:")
    stats.sort_stats(SortKey.CUMULATIVE).print_stats(30)
    
    print("\nProfile sorted by total time:")
    stats.sort_stats(SortKey.TIME).print_stats(30)

if __name__ == "__main__":
    profile_benchmark() 