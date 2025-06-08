"""Example for generating digital signals.

This example demonstrates how to output a finite digital
waveform to different lines on different ports using the DAQ device's internal clock.
"""

import nidaqmx
from nidaqmx.constants import AcquisitionType, LineGrouping
from nidaqmx.stream_writers import DigitalMultiChannelWriter
import numpy as np

with nidaqmx.Task() as task:
    # Create digital patterns for two different ports
    num_samples = 1000
    num_channels = 2  # Two channels: one from port0, one from port1
    
    # Create different patterns for each channel (use uint32 for compatibility)
    data = np.zeros((num_channels, num_samples), dtype=np.uint32)
    
    # Channel 0: Fast toggle pattern (every 10 samples)
    data[0] = [(i // 10) % 2 for i in range(num_samples)]
    
    # Channel 1: Slower toggle pattern (every 50 samples)
    data[1] = [(i // 50) % 2 for i in range(num_samples)]

    # Add digital output lines from different ports
    # First line from port0, second line from port1
    task.do_channels.add_do_chan("PXI1Slot8/port0/line0", line_grouping=LineGrouping.CHAN_PER_LINE)
    task.do_channels.add_do_chan("PXI1Slot8/port1/line0", line_grouping=LineGrouping.CHAN_PER_LINE)
    
    # Configure timing for finite samples
    task.timing.cfg_samp_clk_timing(
        1000.0, sample_mode=AcquisitionType.FINITE, samps_per_chan=num_samples
    )

    # Create the digital multichannel writer
    writer = DigitalMultiChannelWriter(task.out_stream)
    
    # Write data using the multichannel writer
    number_of_samples_written = writer.write_many_sample_port_uint32(data)
    
    print(f"Configured {num_channels} channels on different ports:")
    print(f"  - Port 0, Line 0: Fast toggle (every 10 samples)")
    print(f"  - Port 1, Line 0: Slow toggle (every 50 samples)")
    print(f"Written {number_of_samples_written} samples per channel")
    
    # Start the task
    task.start()
    print("Generating digital patterns on different ports...")
    
    # Wait for completion
    task.wait_until_done()
    task.stop()
    print("Done")