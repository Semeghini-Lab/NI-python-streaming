import sys
import time
import numpy as np

import nidaqmx as ni
from nidaqmx import DaqError
from nidaqmx.constants import AcquisitionType
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

def main():
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

if __name__ == "__main__":
    main()