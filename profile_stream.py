import cProfile
import pstats
from AOCard import AOCard
from Stream import Stream

def main():
    # Create test card with 8 channels
    card = AOCard()
    for i in range(8):
        card.add_channel(i, 0.0)
        # Add some test instructions
        card.channels[i].const(0, 1.0, 1.0)
        card.channels[i].sine(1.0, 1.0, 1000.0, 1.0)
        card.channels[i].linramp(2.0, 1.0, 0.0, 1.0)
        card.channels[i].compile()

    # Create stream with 2M chunk size
    stream = Stream(card, chunk_size=2_000_000, channels_per_process=8)
    
    # Profile 5 chunks
    for _ in range(5):
        stream.calc_next_chunk()

if __name__ == '__main__':
    # Run profiler
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    
    # Save stats to file for snakeviz
    stats = pstats.Stats(profiler)
    stats.dump_stats('stream_profile.prof')
    
    # Print sorted stats
    stats.sort_stats('cumulative').print_stats(30) 