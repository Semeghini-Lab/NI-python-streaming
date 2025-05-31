import sys
import time
import numpy as np

import nidaqmx as ni
from nidaqmx import DaqError
from nidaqmx.constants import AcquisitionType, RegenerationMode

from nidaqmx.stream_writers import AnalogSingleChannelWriter
from nidaqmx.system import System

def list_devices():
    """
    Returns a Python list of all NI-DAQ device names.
    Raises DAQError if no devices or driver is not installed.
    """

    system = System.local()
    devices = system.devices
    if len(devices) == 0:
        raise DaqError("No devices found or NI-DAQmx driver is not installed.")
    return [device.name for device in devices]

def stream_single():
    # 1. Check for available devices
    try:
        devices = list_devices()
    except DaqError:
        print("No devices found or NI-DAQmx driver is not installed.")
        sys.exit(1)

    if not devices:
        print("No devices found.")
        sys.exit(1)

    print(f"Found {len(devices)} device(s):")
    for i, dev in enumerate(devices):
        print(f"{i+1}: {dev.strip()}")

    # Use the PXI1Slot3 device for this test
    device_name = "PXI1Slot3"
    if device_name not in devices:
        print(f"Device {device_name} not found in the list of available devices.")
        sys.exit(1)

    # List all the analog output channels for the selected device
    device = System.local().devices[device_name]
    ao_channels = device.ao_physical_chans
    if not ao_channels:
        print(f"No analog output channels found for device {device_name}.")
        sys.exit(1)
    
    # print(f"Analog output channels for device {device_name}:")
    # for ao_channel in ao_channels:
    #     print(f"  {ao_channel.name}")

    ao_channel = f"{device_name}/ao0"

    # 2. Build a 1 kHz sine wave, 1 second long, 1kS/s sample rate
    sample_rate = 10000  # 100 kS/s
    frequency = 2000.0  # Hz
    duration = 1.0  # seconds
    num_samples = int(sample_rate * duration)

    # Create a sine waveform at 1kHz, sampled at 1kS/s
    t = np.linspace(0, duration, num_samples, endpoint=False)
    data = 2.0* np.sin(2 * np.pi * frequency * t)

    # 3. Create and configure a finite-sample analog-output task
    try:
        with ni.Task() as task:
            # 3.1 Add an analog output channel with +-2V range
            print(f"Configuring task for device {device_name} on channel {ao_channel}...")
            ao = task.ao_channels.add_ao_voltage_chan(
                ao_channel,
                name_to_assign_to_channel="",
                min_val=-4.0, 
                max_val=4.0
            )

            # Print the channel information
            print(f"Channel {ao.name} configured with range [{ao.ao_min}, {ao.ao_max}] V.")

            # 3.2 Configure the sample clock: finite samples, sample_rate S/s, exactly num_samples
            task.timing.cfg_samp_clk_timing(
                rate=sample_rate, 
                sample_mode=AcquisitionType.FINITE, 
                samps_per_chan=num_samples
            )

            # 3.3 Write the entire buffer and do not start the task yet
            num_samples_written = task.write(data, auto_start=False)

            print(f"Written {num_samples_written}/{num_samples} samples to the task.")

            # 3.4 Wait for the input from the user to start the task
            #input("Press Enter to start the task...")

            # 4. Start the task
            task.start()

            # 5. Wait for the task to complete
            task.wait_until_done(timeout=10.0)
            print("Sine wave output completed successfully.")
    except DaqError as e:
        print(f"DAQmx Error: {e}")
        sys.exit(1)


def stream_infinite():
    """
    Stream a continuous sine wave on an analog output channel until stopped by the user.

    This is a simple example of using the NI-DAQmx Python API to stream data continuously.
    """
     # Use the PXI1Slot3 device for this test
    device_name = "PXI1Slot3"
    channel_name = f"{device_name}/ao0"

    # 1. Experiment parameters
    sample_rate = 1.0*1e6  # 1 MS/s
    frequency = 2000.0  # Hz
    amplitude = 2.0  # V
    chunk_size = 4096 # samples per chunk (notify)

    buf_out_size = 4* chunk_size  # total card buffer size in samples
    
    # Sample counter for on-the-fly data generation
    sample_counter = 0

    # 2. Create and configure a continuous analog-output task
    task = ni.Task()

    try:
        # 2.1 Add an analog output channel with the correct amplitude range
        ao = task.ao_channels.add_ao_voltage_chan(
            channel_name,
            name_to_assign_to_channel="",
            min_val=-4.0, 
            max_val=4.0
        )

        print(f"Channel {ao.name} configured with range [{ao.ao_min}, {ao.ao_max}] V.")

        # 2.2 Configure the sample clock: continuous samples, sample_rate S/s
        task.timing.cfg_samp_clk_timing(
            rate=sample_rate, 
            sample_mode=AcquisitionType.CONTINUOUS, 
            samps_per_chan=buf_out_size
        )

        # 2.3 Disable regeneration mode to prevent overwriting data
        task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION

        # 3. Create a writer for the channel and preload the buffer with zeros
        writer = AnalogSingleChannelWriter(task.out_stream, auto_start=False)

        # Preload the buffer with zeros
        initial_data = np.zeros(buf_out_size, dtype=np.float64)
        writer.write_many_sample(initial_data)

        # 4. Register a callback to generate data on-the-fly
        counter = {"counter": sample_counter}

        def every_chunk_samples_callback(task_handle, event_type, n_samples, callback_data):
            """
            Callback function to generate a sine(f(t)*t) wave on-the-fly.
            """
            nonlocal counter

            chunk_counter = counter["counter"]

            # Calculate the initial time offset based on the current counter
            t0 = chunk_counter* chunk_size / sample_rate

            # Calculate the final time offset for the current chunk
            t1 = t0 + chunk_size / sample_rate

            # Generate the time vector for the current chunk
            t = np.linspace(t0, t1, chunk_size, endpoint=False)

            # Generate the frequency as a function of time which oscillates at 1 Hz rate
            freq = frequency #- 1000.0*np.sin(2*2*np.pi*t*0.01)**2

            # Generate the sine wave data for the current chunk
            data = amplitude * np.sin(2 * np.pi * freq * t)

            # Write the generated samples to the output stream
            try:
                writer.write_many_sample(data)
            except DaqError as e:
                print(f"Error writing samples: {e}")
                return 0

            # Update the sample counter
            counter["counter"] += 1

            return 0
        
        # Register the callback function
        task.register_every_n_samples_transferred_from_buffer_event(
            chunk_size,
            every_chunk_samples_callback
        )

        # 5. Start the task
        task.start()

        # Wait for the user to press Enter to stop the task
        input("Streaming started. Press ENTER to stop...")

        # 6. Stop the task
        task.stop()

        # 7. Clean up
        task.close()

    except DaqError as e:
        print(f"DAQmx Error: {e}")
        task.close()
        sys.exit(1)

if __name__ == "__main__":
    stream_infinite()