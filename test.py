
"""Test for single digital output channel streaming using task.write()"""

import sys
import time
import numpy as np
import nidaqmx as ni
from nidaqmx.constants import AcquisitionType, LineGrouping, RegenerationMode
from nidaqmx.errors import DaqError

device_name = "PXI1Slot8"  # Match the SequenceStreamer device
channel_name = f"{device_name}/port0/line0"  # Single channel

sample_rate = 10_000_000
frequency = 100.0   # Hz square wave for the single channel
chunk_size = 65536
buf_out_size = 2 * chunk_size  # buffer size
sample_counter = 0


task = ni.Task()

try:
    print(f"Configuring single-channel digital streaming on {device_name}...")
    print(f"DEBUG: Channel: {channel_name}")
    print(f"DEBUG: Sample rate: {sample_rate} Hz")
    print(f"DEBUG: Square wave frequency: {frequency} Hz")
    print(f"DEBUG: Chunk size: {chunk_size} samples")
    print(f"DEBUG: Buffer size: {buf_out_size} samples")

    # Add single digital output channel
    task.do_channels.add_do_chan(channel_name, line_grouping=LineGrouping.CHAN_PER_LINE)
    print(f"DEBUG: Added digital output channel: {channel_name}")
    
    print(f"Total channels configured: {len(task.channels)}")

    # Configure buffered sample clock
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=buf_out_size
    )
    print(f"DEBUG: Configured sample clock timing")

    # Disable regeneration to avoid repeating stale data
    task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION
    print(f"DEBUG: Disabled regeneration mode")

    # Preload buffer with zeros for single channel (1D array)
    preload_data = np.zeros(buf_out_size, dtype=bool)  # 1D array for single channel
    task.write(preload_data, auto_start=False)
    print(f"DEBUG: Preloaded buffer with shape: {preload_data.shape}")

    # Start the task
    task.start()
    print(f"Streaming started on {channel_name} at {sample_rate} Hz")
    print(f"DEBUG: Task started successfully")

    # Start streaming loop
    period_samples = int(sample_rate / frequency)
    start_time = time.time()
    
    print(f"DEBUG: Starting streaming loop")
    print(f"DEBUG: period_samples={period_samples}")
    print(f"DEBUG: Chunk size: {chunk_size}, Buffer size: {buf_out_size}")
    print(f"DEBUG: Square wave period: {period_samples/sample_rate:.6f} seconds")

    while True:
        # Check available space in buffer
        space = task.out_stream.space_avail
        
        # Print buffer status every 100 iterations
        if sample_counter % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"DEBUG: Buffer space: {space}/{buf_out_size}, Sample counter: {sample_counter}, Elapsed: {elapsed_time:.2f}s")

        if space >= chunk_size:
            # Generate square wave pattern for single channel
            indices = np.arange(chunk_size) + sample_counter * chunk_size
            
            # Generate pattern for the single channel
            pattern = ((indices % period_samples) < (period_samples // 2)).astype(bool)
            
            # For single digital line, use 1D array directly
            data = pattern

            # Debug first few samples to verify pattern generation
            if sample_counter < 3:
                print(f"DEBUG: Sample counter {sample_counter}, data shape: {data.shape}")
                print(f"DEBUG: First 10 values: {data[:10]}")

            task.write(data, auto_start=False)
            
            if sample_counter % 100 == 0:
                print(f"DEBUG: Successfully wrote chunk {sample_counter}")

            sample_counter += 1

        else:
            # Sleep a short time to avoid busy-wait
            if sample_counter < 10:  # Only print for first few iterations
                print(f"DEBUG: Waiting for buffer space... current space: {space}, need: {chunk_size}")
            time.sleep(0.001)

except KeyboardInterrupt:
    print("\nStopped by user.")

except DaqError as e:
    print(f"DAQ Error: {e}")
    sys.exit(1)

except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)

finally:
    print("Stopping and closing task...")
    try:
        task.stop()
        task.close()
    except:
        pass
    print("Task cleaned up successfully.")