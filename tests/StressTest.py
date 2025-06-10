# StressTest.py
# Stress test for the NI streaming system with multiple cards and channels
# Tests PXI1Slot3,4 (analog) and PXI1Slot8,9 (digital) with random sequences

import sys
import numpy as np
import random
from Sequences import AOSequence, DOSequence
from NICard import NICard
from SequenceStreamer import SequenceStreamer

def generate_random_analog_sequence(channel_id: str, sample_rate: int, total_duration: float = 60.0):
    """
    Generate a random analog sequence with various instruction types.
    
    Args:
        channel_id: Channel identifier (e.g., "ao0")
        sample_rate: Sample rate in Hz
        total_duration: Total sequence duration in seconds
    
    Returns:
        AOSequence with random instructions
    """
    seq = AOSequence(channel_id=channel_id, sample_rate=sample_rate)
    
    current_sample = 0
    total_samples = int(total_duration * sample_rate)
    last_value = 0.0
    
    # Generate random instructions until we reach total_duration
    while current_sample < total_samples:
        # Random instruction duration in samples (between 1ms and 1s worth of samples)
        min_samples = max(1, int(0.001 * sample_rate))  # 1ms minimum
        max_samples = min(int(1.0 * sample_rate), total_samples - current_sample)  # 1s maximum
        
        if max_samples <= min_samples:
            # Not enough samples left for a meaningful instruction
            break
            
        duration_samples = random.randint(min_samples, max_samples)
        
        # Convert to time for the instruction
        start_time = current_sample / sample_rate
        duration_time = duration_samples / sample_rate
        
        # Random instruction type
        instruction_type = random.choice(['const', 'linramp', 'sine'])
        
        if instruction_type == 'const':
            # Random constant value between -5V and +5V
            value = random.uniform(-5.0, 5.0)
            seq.const(start_time, duration_time, value=value)
            last_value = value
            
        elif instruction_type == 'linramp':
            # Random start and end values
            start_value = last_value if random.random() > 0.5 else random.uniform(-5.0, 5.0)
            end_value = random.uniform(-5.0, 5.0)
            seq.linramp(start_time, duration_time, start=start_value, end=end_value)
            last_value = end_value
            
        elif instruction_type == 'sine':
            # Random sine parameters
            freq = random.uniform(1, 10000)  # 1 Hz to 10 kHz
            amp = random.uniform(0.1, 3.0)   # 0.1V to 3V amplitude
            phase = random.uniform(0, 2 * np.pi)
            seq.sine(start_time, duration_time, freq=freq, amp=amp, phase=phase)
            # For sine, we don't know the exact end value, so use a random one
            last_value = random.uniform(-amp, amp)
        
        current_sample += duration_samples
    
    print(f"Generated analog sequence for {channel_id}: {len(seq.instructions)} instructions over {total_duration}s")
    return seq

def generate_random_digital_sequence(channel_id: str, sample_rate: int, total_duration: float = 60.0):
    """
    Generate a random digital sequence with high/low patterns.
    
    Args:
        channel_id: Channel identifier (e.g., "port0/line0")
        sample_rate: Sample rate in Hz
        total_duration: Total sequence duration in seconds
    
    Returns:
        DOSequence with random instructions
    """
    seq = DOSequence(channel_id=channel_id, sample_rate=sample_rate)
    
    current_sample = 0
    total_samples = int(total_duration * sample_rate)
    current_state = random.choice([True, False])  # Start with random state
    
    # Generate random high/low patterns
    while current_sample < total_samples:
        # Random duration in samples (between 0.0001s and 0.5s worth of samples)
        min_samples = max(1, int(0.0001 * sample_rate))  # 0.1ms minimum
        max_samples = min(int(0.5 * sample_rate), total_samples - current_sample)  # 0.5s maximum
        
        if max_samples <= min_samples:
            # Not enough samples left for a meaningful instruction
            break
            
        duration_samples = random.randint(min_samples, max_samples)
        
        # Convert to time for the instruction
        start_time = current_sample / sample_rate
        duration_time = duration_samples / sample_rate
        
        if current_state:
            seq.high(start_time, duration_time)
        else:
            seq.low(start_time, duration_time)
        
        current_sample += duration_samples
        
        # Toggle state for next instruction
        current_state = not current_state
    
    print(f"Generated digital sequence for {channel_id}: {len(seq.instructions)} instructions over {total_duration}s")
    return seq

def create_stress_test(scale="medium"):
    """
    Create a comprehensive stress test with multiple cards and channels.
    
    Args:
        scale: "small" (32 channels), "medium" (64 channels), "large" (128 channels)
    """
    print("=== NI Streaming System Stress Test ===")
    print(f"Scale: {scale.upper()}")
    print("Initializing cards and generating random sequences...")
    
    # Set random seed for reproducible tests (remove this line for truly random tests)
    random.seed(42)
    np.random.seed(42)
    
    # Configuration
    analog_sample_rate = 400_000  
    digital_sample_rate = 10_000_000  # 10 MHz for digital
    total_duration = 10.0  # 1 minute
    chunk_size = 65536#*4
    
    # Scale configuration
    if scale == "small":
        analog_channels_per_card = 4
        digital_ports_per_card = 1  # 8 channels total per card
    elif scale == "medium":
        analog_channels_per_card = 8  
        digital_ports_per_card = 1  # 8 channels total per card
    elif scale == "large":
        analog_channels_per_card = 32
        digital_ports_per_card = 4  # 32 channels total per card
    else:
        raise ValueError(f"Unknown scale: {scale}")
    
    print(f"Configuration: {analog_channels_per_card} analog channels per card, {digital_ports_per_card*8} digital channels per card")
    
    # === ANALOG CARDS ===
    # PXI1Slot3 - Analog card
    analog_channels_card1 = []
    for ch_num in range(analog_channels_per_card):
        channel_id = f"ao{ch_num}"
        seq = generate_random_analog_sequence(channel_id, analog_sample_rate, total_duration)
        analog_channels_card1.append(seq)
    
    card1 = NICard(
        device_name="PXI1Slot3",
        sample_rate=analog_sample_rate,
        sequences=analog_channels_card1,
        is_primary=True,
        trigger_source="PXI_Trig0",
        clock_source="PXI_Trig7"
    )
    
    # PXI1Slot4 - Analog card
    analog_channels_card2 = []
    for ch_num in range(analog_channels_per_card):
        channel_id = f"ao{ch_num}"
        seq = generate_random_analog_sequence(channel_id, analog_sample_rate, total_duration)
        analog_channels_card2.append(seq)
    
    card2 = NICard(
        device_name="PXI1Slot4",
        sample_rate=analog_sample_rate,
        sequences=analog_channels_card2,
        trigger_source=card1.trigger_source,
        clock_source=card1.clock_source
    )
    
    # === DIGITAL CARDS ===
    # PXI1Slot8 - Digital card
    digital_channels_card1 = []
    for port_num in range(digital_ports_per_card):
        for line_num in range(8):  # line0 through line7
            channel_id = f"port{port_num}/line{line_num}"
            seq = generate_random_digital_sequence(channel_id, digital_sample_rate, total_duration)
            digital_channels_card1.append(seq)
    
    card3 = NICard(
        device_name="PXI1Slot8",
        sample_rate=digital_sample_rate,
        sequences=digital_channels_card1,
        trigger_source=card1.trigger_source,
        clock_source=card1.clock_source
    )
    
    # PXI1Slot9 - Digital card
    digital_channels_card2 = []
    for port_num in range(digital_ports_per_card):
        for line_num in range(8):  # line0 through line7
            channel_id = f"port{port_num}/line{line_num}"
            seq = generate_random_digital_sequence(channel_id, digital_sample_rate, total_duration)
            digital_channels_card2.append(seq)
    
    card4 = NICard(
        device_name="PXI1Slot9",
        sample_rate=digital_sample_rate,
        sequences=digital_channels_card2,
        trigger_source=card1.trigger_source,
        clock_source=card1.clock_source
    )
    
    # Aggregate all cards
    cards = [card1, card2, card3, card4]
    
    # Print summary
    print(f"\n=== Test Configuration ===")
    print(f"Total duration: {total_duration} seconds")
    print(f"Analog sample rate: {analog_sample_rate:,} Hz")
    print(f"Digital sample rate: {digital_sample_rate:,} Hz")
    print(f"Chunk size: {chunk_size:,} samples")
    print(f"Total cards: {len(cards)}")
    print(f"Total channels: {sum(len(card.sequences) for card in cards)}")
    
    for i, card in enumerate(cards):
        card_type = "Analog" if not card.sequences[0].__class__.__name__.startswith('DO') else "Digital"
        print(f"  Card {i+1} ({card.device_name}): {len(card.sequences)} {card_type.lower()} channels")
    
    # Compile all cards
    print(f"\n=== Compiling Sequences ===")
    for i, card in enumerate(cards):
        print(f"Compiling card {i+1} ({card.device_name})...")
        card.compile(chunk_size=chunk_size, external_stop_time=total_duration)
        print(f"  Card {i+1}: {card.num_chunks} chunks to stream")
    
    total_chunks = sum(card.num_chunks for card in cards)
    print(f"Total chunks across all cards: {total_chunks:,}")
    
    # Run the stress test
    print(f"\n=== Starting Stress Test ===")
    print("This will stream random sequences to all channels for 60 seconds...")
    print("Press Ctrl+C to stop early if needed.")
    
    try:
        with SequenceStreamer(
            cards=cards,
            num_workers=4,  # Use more workers for stress test
            num_writers=4,  # Use more writers for stress test
            pool_size=16,   # Larger pool size for stress test
        ) as streamer:
            streamer.start()
            
        print(f"\n=== Stress Test Completed Successfully! ===")
        print("All sequences streamed without errors.")
        
    except KeyboardInterrupt:
        print(f"\n=== Stress Test Interrupted by User ===")
        print("Test stopped early.")
        
    except Exception as e:
        print(f"\n=== Stress Test Failed ===")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    # Allow specifying scale from command line
    scale = "medium"  # default
    if len(sys.argv) > 1:
        scale = sys.argv[1].lower()
        if scale not in ["small", "medium", "large"]:
            print("Usage: python StressTest.py [small|medium|large]")
            print("  small:  32 total channels (4+4 analog, 8+8 digital)")
            print("  medium: 32 total channels (8+8 analog, 8+8 digital)")  
            print("  large:  128 total channels (32+32 analog, 32+32 digital)")
            sys.exit(1)
    
    create_stress_test(scale) 