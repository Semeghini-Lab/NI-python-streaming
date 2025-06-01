# Writer.py
# Created by: Marcin Kalinowski & Yi Zhu

import os
import time
import struct
import select
import numpy as np
from multiprocessing import shared_memory

import nidaqmx as ni
from nidaqmx.constants import AcquisitionType, RegenerationMode

class UnderrunError(Exception):
    """Custom exception for underrun errors in the DAQ device."""
    pass

class Writer:
    """
    Writer process: reads chunk assignments via a pipe, writes data to the DAQ device,
    and reports results back via another pipe.
    """

    READY_STRUCT = struct.Struct('>I I')  # (buf_idx)
    REPORT_STRUCT = struct.Struct('>I')  # (buf_idx)

    def __ini__(
            self,
            ready_r_fd: int,  # Read file descriptor for ready signal
            report_w_fd: int,  # Write file descriptor for free signal
            shm_name: str,  # Name of the shared memory segment
            sample_rate: float,  # Sample rate in Hz
            num_channels: int,  # Total number of channels
            chunk_size: int,  # Size of each chunk in samples
            outbuf_num_chunks: int,  # Total number of chunks in the card buffer
            pool_size: int,  # Size of the memory pool for calculations
            device_name: str,  # Name of the DAQ device
            ao_channels: list,  # List of analog output channel named tuples
            do_channels: list,  # List of digital output channels named tuples

    ) -> None:
        self.ready_r_fd = ready_r_fd
        self.report_w_fd = report_w_fd
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.outbuf_num_chunks = outbuf_num_chunks
        self.pool_size = pool_size
        self.num_channels = num_channels

        # Attach to shared memory segment
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.buffer = np.ndarray((pool_size, num_channels, chunk_size), dtype=np.float64, buffer=self.shm.buf)

        # Save channel and device information
        self.device_name = device_name
        self.ao_channels = ao_channels
        self.do_channels = do_channels

        assert len(ao_channels) + len(do_channels) == num_channels, \
            f"Total number of channels ({len(ao_channels) + len(do_channels)}) does not match shared memory size ({num_channels})."
        
        # Initialize the DAQ device
        self._configure_device()

        # Non-blocking mode for the ready pipe
        os.set_blocking(self.ready_r_fd, False)

        # Initialize the sequence counter for continuity tracking (sanity check)
        self.last_seq_written = -1  # Last sequence number written to the device

    def _configure_device(self):
        self.task = ni.Task()
        # Add analog output channels
        for ao in self.ao_channels:
            self.task.ao_channels.add_ao_voltage_chan(
                ao.name,
                min_val=ao.min_val,
                max_val=ao.max_val
            )

        # Add digital output channels
        # TODO: Implement digital output channel configuration

        # Calculate the total output buffer size
        outbuf_size = self.chunk_size * self.outbuf_num_chunks

        # Configure the sample clock
        self.task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=outbuf_size
        )

        # Set regeneration mode
        self.task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION

        # Create a writer for the analog output channels
        self.writer = ni.stream_writers.AnalogMultiChannelWriter(
            self.task.out_stream,
            auto_start=False
        )

    def _every_chunk_samples_callback(self, task_handle, event_type, n_samples, callback_data):
        """
        Callback function to write data to the DAQ device.
        This is called every time a chunk of samples is transferred from the buffer.
        """
        # Wait for the ready signal from the pipe up to timeout
        rlist, _, _ = select.select([self.ready_r_fd], [], [], self.timeout)
        if not rlist:
            raise UnderrunError(f"Timed out after {self.timeout*1e3}ms waiting for ready signal from the pipe.")
        
        # If we got a signal, read it
        read_msg = os.read(self.ready_r_fd, self.READY_STRUCT.size)
        if len(read_msg) < self.READY_STRUCT.size:
            raise ValueError("Received incomplete ready signal data.")
        
        # Unpack the ready signal
        seq, buf_idx = self.READY_STRUCT.unpack(read_msg)

        # Make sure the sequence number is as expected
        if seq != self.last_seq_written + 1:
            raise ValueError(f"Out-of-order error: streaming chunk {seq}, expected {self.last_seq_written + 1}.")
        
        # Get the flattened view of the current chunk memory
        flat_buffer = self.buffer[buf_idx].reshape(-1)

        # Send the data to the device
        try:
            self.writer.write_many_sample(flat_buffer)                                                                                                                                    
        except ni.errors.DaqError as e:
            print(f"DAQ error: {e}")
            raise

        # Update the last sequence number written
        self.last_seq_written = seq

        # Report back that the buffer is free
        os.write(self.report_w_fd, self.REPORT_STRUCT.pack(buf_idx))

    def _initialize_device(self, timeout=1.0):
        """Load the device buffer until full."""
        realtime_timeout = self.timeout

        # Set longer timeout for initial buffer loading
        self.timeout = timeout

        # Preload the buffer
        for i in range(self.outbuf_num_chunks):
            self._every_chunk_samples_callback(None, None, None, None)

        # Reset the timeout to the original value
        self.timeout = realtime_timeout

    def run(self):
        # Preload the buffer before starting the task
        self._initialize_device()

        # Register the callback
        self.task.register_every_n_samples_transferred_from_buffer_event(
            every_n_samples=self.chunk_size,
            callback=self._every_chunk_samples_callback
        )
        
        # Start the task
        self.task.start()

        # Wait for the interrupt
        try:
            while True:
                time.sleep(0.1)
        except UnderrunError:
            print("Underrun error occurred. Stopping the task.")
        except KeyboardInterrupt:
            print("Keyboard interrupt received. Stopping the task.")
        finally:
            # Stop the task gracefully
            self.task.stop()
            self.task.close()