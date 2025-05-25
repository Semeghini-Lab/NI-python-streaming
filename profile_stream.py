import cProfile
import pstats
import numpy as np
from AOCard import AOCard
from Stream import Stream

def create_test_card():
    """Create a test card with 16 channels."""
    frequencies = np.logspace(np.log10(1e3), np.log10(200e3), 16).tolist()
    card = AOCard(samp_rate=10e6)
    
    for i, freq in enumerate(frequencies):
        ch = card.add_channel(i, default_val=0.0)
        ch.sine(0, 2.0, freq, 1.5, offset=0, phase=0)
        ch.compile()
    
    return card

def main():
    # Create test card
    card = create_test_card()
    
    # Create stream with optimal configuration
    stream = Stream(card, chunk_size=2_000_000, channels_per_process=4)
    
    # Process 10 chunks
    for _ in range(10):
        stream.calc_next_chunk()

if __name__ == '__main__':
    # Run profiler
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    
    # Print sorted stats
    stats = pstats.Stats(profiler)
    print("\nTop 20 functions by cumulative time:")
    stats.sort_stats('cumulative').print_stats(20)
    
    print("\nTop 20 functions by total time:")
    stats.sort_stats('time').print_stats(20) 