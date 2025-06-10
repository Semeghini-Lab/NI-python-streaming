import sys
import time
import numpy as np

import nidaqmx as ni
from nidaqmx import DaqError
from nidaqmx.constants import AcquisitionType, RegenerationMode

from nidaqmx.stream_writers import AnalogSingleChannelWriter, DigitalSingleChannelWriter
from nidaqmx.system import System

from nidaqmx.constants import LineGrouping

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

def stream_infinite_do():
    """
    Stream continuous digital patterns on a digital output channel until stopped by the user.

    This is a digital equivalent of stream_infinite, using NI-DAQmx digital I/O functions
    to stream boolean patterns continuously to a digital output port.
    
    SAFE VERSION: Uses conservative parameters to prevent system crashes.
    """
    # Use the PXI1Slot8 device (6535 digital card) for this test
    device_name = "PXI1Slot8"
    channel_name = f"{device_name}/port0"  # Use entire port0 for 8-line port

    # 1. Experiment parameters - CONSERVATIVE SETTINGS
    sample_rate = 10_000_000 
    frequency = 100.0  # Hz - square wave frequency (reduced)
    chunk_size = 1000  # samples per chunk (reduced from 4096)
    buf_out_size = 2 * chunk_size  # total card buffer size (reduced from 4x)
    
    # Sample counter for on-the-fly data generation
    sample_counter = 0

    # 2. Create and configure a continuous digital-output task
    task = ni.Task()

    try:
        print(f"Configuring digital streaming on {device_name}...")
        
        # 2.1 Add a digital output channel for the entire port
        do = task.do_channels.add_do_chan(channel_name)
        print(f"Digital channel {do.name} configured.")

        # 2.2 Configure the sample clock: continuous samples, sample_rate S/s
        task.timing.cfg_samp_clk_timing(
            rate=sample_rate, 
            sample_mode=AcquisitionType.CONTINUOUS, 
            samps_per_chan=buf_out_size
        )

        # 2.3 Disable regeneration mode to prevent overwriting data
        task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION

        # 3. Preload the buffer with zeros using direct task write method
        # Use uint32 format for port writing (1D array for single port)
        initial_data = np.zeros(buf_out_size, dtype=np.uint32)
        task.write(initial_data, auto_start=False)

        # # 4. Register a callback to generate digital patterns on-the-fly
        # counter = {"counter": sample_counter}

        def every_chunk_samples_callback(task_handle, event_type, n_samples, callback_data):
            print(f"Callback called with {n_samples} samples")
            pass
        #     """
        #     Callback function to generate digital square wave patterns on-the-fly.
        #     SAFE VERSION: Simplified processing to prevent system overload.
        #     """
        #     nonlocal counter

        #     try:
        #         chunk_counter = counter["counter"]

        #         # Simplified time calculation
        #         t_start = chunk_counter * chunk_size / sample_rate
                
        #         # Pre-calculate the period in samples to avoid repeated calculations
        #         period_samples = int(sample_rate / frequency)
                
        #         # Generate simple square wave pattern without complex math
        #         sample_indices = np.arange(chunk_size)
        #         global_sample_indices = sample_indices + chunk_counter * chunk_size
                
        #         # Simple square wave: True for first half of period, False for second half
        #         pattern_line0 = ((global_sample_indices % period_samples) < (period_samples // 2)).astype(np.uint32)
        #         pattern_line1 = ((global_sample_indices % (period_samples // 2)) < (period_samples // 4)).astype(np.uint32)
                
        #         # Combine patterns into uint32 port data
        #         # Each bit represents a line: bit 0 = line 0, bit 1 = line 1, etc.
        #         port_data = (pattern_line0 * 1) + (pattern_line1 * 2)  # bits 0 and 1
                
        #         # Use 1D array for single port direct write
        #         data = port_data.astype(np.uint32)

        #         # Write the generated samples to the output stream using direct task write
        #         task.write(data, auto_start=False)

        #         # Update the sample counter
        #         counter["counter"] += 1

        # #         return 0

        #     except Exception as e:
        #         print(f"Error in callback: {e}")
        #         return -1  # Signal error to stop the task
        
        # Register the callback function
        task.register_every_n_samples_transferred_from_buffer_event(
            chunk_size-10,
            every_chunk_samples_callback
        )

        # 5. Start the task
        print(f"Starting digital streaming on {channel_name}...")
        print(f"Sample rate: {sample_rate} Hz, Chunk size: {chunk_size}")
        print(f"Square wave frequency: {frequency} Hz")
        print("SAFE MODE: Conservative parameters to prevent system crashes")
        task.start()

        # Wait for the user to press Enter to stop the task
        try:
            input("Streaming started. Press ENTER to stop...")
        except KeyboardInterrupt:
            print("\nStopping due to Ctrl+C...")

        # 6. Stop the task
        print("Stopping digital streaming...")
        task.stop()

        # 7. Clean up
        task.close()

        print("Digital streaming completed successfully.")

    except DaqError as e:
        print(f"DAQmx Error: {e}")
        if task:
            try:
                task.stop()
                task.close()
            except:
                pass
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if task:
            try:
                task.stop()
                task.close()
            except:
                pass
        sys.exit(1)

def stream_infinite_do_polling():
    """
    Stream continuous digital patterns on digital output channels using polling
    to detect when to feed more data into the buffer.
    
    This version avoids callbacks, suitable for hardware like NI-6535 in buffered mode.
    Outputs different patterns on 4 digital lines from different ports.
    """
    device_name = "PXI1Slot8"
    channel_names = [
        f"{device_name}/port0/line0", 
        f"{device_name}/port0/line2",
        f"{device_name}/port1/line1", 
        f"{device_name}/port1/line3"
    ]

    sample_rate = 10_000_000
    frequency_ch0 = 100.0   # Hz square wave for port0/line0
    frequency_ch1 = 250.0   # Hz square wave for port0/line2
    frequency_ch2 = 400.0   # Hz square wave for port1/line1
    frequency_ch3 = 600.0   # Hz square wave for port1/line3
    chunk_size = 65536
    buf_out_size = 2 * chunk_size  # buffer size
    sample_counter = 0
    

    task = ni.Task()

    try:
        print(f"Configuring quad-channel digital streaming on {device_name}...")
        print(f"DEBUG: Channels: {channel_names}")
        print(f"DEBUG: Sample rate: {sample_rate} Hz")
        print(f"DEBUG: CH0 - Square wave frequency: {frequency_ch0} Hz (port0/line0)")
        print(f"DEBUG: CH1 - Square wave frequency: {frequency_ch1} Hz (port0/line2)")
        print(f"DEBUG: CH2 - Square wave frequency: {frequency_ch2} Hz (port1/line1)")
        print(f"DEBUG: CH3 - Square wave frequency: {frequency_ch3} Hz (port1/line3)")
        print(f"DEBUG: Chunk size: {chunk_size} samples")
        print(f"DEBUG: Buffer size: {buf_out_size} samples")

        # Add digital output channels
        for i, channel_name in enumerate(channel_names):
            task.do_channels.add_do_chan(channel_name, line_grouping=LineGrouping.CHAN_PER_LINE)
            print(f"DEBUG: Added digital output channel {i}: {channel_name}")
        
        print(f"Total channels configured: {len(channel_names)}")

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

        # Preload buffer with zeros for all channels
        preload_data = np.zeros((4, buf_out_size), dtype=bool)  # 4 channels x buf_out_size samples
        task.write(preload_data, auto_start=False)
        print(f"DEBUG: Preloaded buffer with shape: {preload_data.shape}")

        # Start the task
        task.start()
        print(f"Streaming started on {channel_names} at {sample_rate} Hz")
        print(f"DEBUG: Task started successfully")

        # Start streaming loop
        period_samples_ch0 = int(sample_rate / frequency_ch0)
        period_samples_ch1 = int(sample_rate / frequency_ch1)
        period_samples_ch2 = int(sample_rate / frequency_ch2)
        period_samples_ch3 = int(sample_rate / frequency_ch3)
        start_time = time.time()
        
        print(f"DEBUG: Starting streaming loop")
        print(f"DEBUG: CH0 period_samples={period_samples_ch0}, CH1 period_samples={period_samples_ch1}")
        print(f"DEBUG: CH2 period_samples={period_samples_ch2}, CH3 period_samples={period_samples_ch3}")
        print(f"DEBUG: Chunk size: {chunk_size}, Buffer size: {buf_out_size}")
        print(f"DEBUG: CH0 square wave period: {period_samples_ch0/sample_rate:.6f} seconds")
        print(f"DEBUG: CH1 square wave period: {period_samples_ch1/sample_rate:.6f} seconds")
        print(f"DEBUG: CH2 square wave period: {period_samples_ch2/sample_rate:.6f} seconds")
        print(f"DEBUG: CH3 square wave period: {period_samples_ch3/sample_rate:.6f} seconds")

        while True:
            # Check available space in buffer
            space = task.out_stream.space_avail
            
            # Print buffer status every 100 iterations
            if sample_counter % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"DEBUG: Buffer space: {space}/{buf_out_size}, Sample counter: {sample_counter}, Elapsed: {elapsed_time:.2f}s")

            if space >= chunk_size:
                # Generate square wave patterns for all channels
                indices = np.arange(chunk_size) + sample_counter * chunk_size
                
                # Generate pattern for channel 0 (port0/line0)
                pattern_ch0 = ((indices % period_samples_ch0) < (period_samples_ch0 // 2)).astype(bool)
                
                # Generate pattern for channel 1 (port0/line2)  
                pattern_ch1 = ((indices % period_samples_ch1) < (period_samples_ch1 // 2)).astype(bool)
                
                # Generate pattern for channel 2 (port1/line1)
                pattern_ch2 = ((indices % period_samples_ch2) < (period_samples_ch2 // 2)).astype(bool)
                
                # Generate pattern for channel 3 (port1/line3)
                pattern_ch3 = ((indices % period_samples_ch3) < (period_samples_ch3 // 2)).astype(bool)
                
                # Combine patterns into 2D array: [channel, samples]
                data = np.array([pattern_ch0, pattern_ch1, pattern_ch2, pattern_ch3], dtype=bool)

                # Debug first few samples to verify pattern generation
                if sample_counter < 3:
                    print(f"DEBUG: Sample counter {sample_counter}, data shape: {data.shape}")
                    print(f"DEBUG: CH0 first 10 values: {data[0, :10]}")
                    print(f"DEBUG: CH1 first 10 values: {data[1, :10]}")
                    print(f"DEBUG: CH2 first 10 values: {data[2, :10]}")
                    print(f"DEBUG: CH3 first 10 values: {data[3, :10]}")

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

def stream_infinite_ao_polling():
    """
    Stream continuous analog output using polling to detect when to feed more data into the buffer.
    
    This version avoids callbacks and uses polling to check buffer availability.
    Outputs different waveforms on 2 analog output channels.
    """
    device_name = "PXI1Slot3"
    channel_names = [f"{device_name}/ao0", f"{device_name}/ao1"]

    sample_rate = 1_000_000  # 1 MS/s
    frequency_ch0 = 2000.0  # Hz sine wave for channel 0
    frequency_ch1 = 1500.0  # Hz sine wave for channel 1
    amplitude_ch0 = 2.0  # V for channel 0
    amplitude_ch1 = 1.5  # V for channel 1
    chunk_size = 65536
    buf_out_size = 2 * chunk_size  # buffer size
    sample_counter = 0

    task = ni.Task()

    try:
        print(f"Configuring dual-channel analog streaming on {device_name}...")
        print(f"DEBUG: Channels: {channel_names}")
        print(f"DEBUG: Sample rate: {sample_rate} Hz")
        print(f"DEBUG: CH0 - Sine wave frequency: {frequency_ch0} Hz, Amplitude: {amplitude_ch0} V")
        print(f"DEBUG: CH1 - Sine wave frequency: {frequency_ch1} Hz, Amplitude: {amplitude_ch1} V")
        print(f"DEBUG: Chunk size: {chunk_size} samples")
        print(f"DEBUG: Buffer size: {buf_out_size} samples")

        # Add analog output channels
        ao_channels = []
        for i, channel_name in enumerate(channel_names):
            ao = task.ao_channels.add_ao_voltage_chan(
                channel_name,
                name_to_assign_to_channel="",
                min_val=-4.0,
                max_val=4.0
            )
            ao_channels.append(ao)
            print(f"DEBUG: Added analog output channel {i}: {channel_name}")
            print(f"Channel {ao.name} configured with range [{ao.ao_min}, {ao.ao_max}] V.")
        
        print(f"Total channels configured: {len(ao_channels)}")

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

        # Preload buffer with zeros for both channels
        preload_data = np.zeros((2, buf_out_size), dtype=np.float64)  # 2 channels x buf_out_size samples
        task.write(preload_data, auto_start=False)
        print(f"DEBUG: Preloaded buffer with {preload_data.shape} zeros")

        # Start the task
        task.start()
        print(f"Streaming started on {channel_names} at {sample_rate} Hz")
        print(f"DEBUG: Task started successfully")

        # Start streaming loop
        start_time = time.time()
        
        print(f"DEBUG: Starting streaming loop")
        print(f"DEBUG: Chunk size: {chunk_size}, Buffer size: {buf_out_size}")
        print(f"DEBUG: CH0 sine wave period: {1.0/frequency_ch0:.6f} seconds")
        print(f"DEBUG: CH1 sine wave period: {1.0/frequency_ch1:.6f} seconds")

        while True:
            # Check available space in buffer
            space = task.out_stream.space_avail
            
            # Print buffer status every 100 iterations
            if sample_counter % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"DEBUG: Buffer space: {space}/{buf_out_size}, Sample counter: {sample_counter}, Elapsed: {elapsed_time:.2f}s")

            if space >= chunk_size:
                # Generate sine wave for this chunk
                # Calculate the initial time offset based on the current counter
                t0 = sample_counter * chunk_size / sample_rate
                t1 = t0 + chunk_size / sample_rate
                
                # Generate the time vector for the current chunk
                t = np.linspace(t0, t1, chunk_size, endpoint=False)
                
                # Generate the sine wave data for both channels
                data_ch0 = amplitude_ch0 * np.sin(2 * np.pi * frequency_ch0 * t)
                data_ch1 = amplitude_ch1 * np.sin(2 * np.pi * frequency_ch1 * t)
                
                # Combine into 2D array: [channel, samples]
                data = np.array([data_ch0, data_ch1], dtype=np.float64)

                # Debug first few samples to verify pattern generation
                if sample_counter < 3:
                    print(f"DEBUG: Sample counter {sample_counter}, data shape: {data.shape}")
                    print(f"DEBUG: CH0 first 10 values: {data[0, :10]}")
                    print(f"DEBUG: CH1 first 10 values: {data[1, :10]}")
                    print(f"DEBUG: Time range: {t0:.6f} to {t1:.6f} seconds")

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



if __name__ == "__main__":
    # stream_infinite()
    stream_infinite_do_polling()