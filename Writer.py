# Writer.py
# Created by: Marcin Kalinowski & Yi Zhu

import os
import time
import struct
import select
import numpy as np
from AOSequence import AOSequence
from multiprocessing import Process, shared_memory

TEST_MODE = True

class UnderrunError(Exception):
    """Custom exception for underrun errors in the DAQ device."""
    pass

class Writer(Process):
    """
    Writer process: reads chunk assignments via a pipe, writes data to the DAQ device,
    and reports results back via another pipe.
    """

    READY_STRUCT = struct.Struct('>q i')  # (chunk_idx, buf_idx)
    REPORT_STRUCT = struct.Struct('>q i')  # (chunk_idx, buf_idx)

    def __init__(
            self,
            shm_name: str,  # Name of the shared memory segment
            sample_rate: float,  # Sample rate in Hz
            chunk_size: int,  # Size of each chunk in samples
            outbuf_num_chunks: int,  # Total number of chunks in the card buffer
            pool_size: int,  # Size of the memory pool for calculations
            device_name: str,  # Name of the DAQ device
            ao_channels: list[AOSequence],  # List of analog output channel named tuples
            do_channels: list,  # List of digital output channels named tuples
            timeout: float = 100*1e-6,  # Grace period after next chunk is expected
    ) -> None:
        super().__init__(name="Writer")
        self.shm_name = shm_name
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.outbuf_num_chunks = outbuf_num_chunks
        self.pool_size = pool_size
        self.device_name = device_name
        self.ao_channels = ao_channels
        self.do_channels = do_channels
        self.num_ao_channels = len(ao_channels)
        self.num_do_channels = len(do_channels)
        self.timeout = timeout
        self.running = True

        # Create all communication pipes in parent
        self.ready_r, self.ready_w = os.pipe()
        self.report_r, self.report_w = os.pipe()

        # These will be set in run()
        self.shm = None
        self.buffer = None
        self.task = None
        self.writer = None

    def run(self):
        if not TEST_MODE:
            import nidaqmx as ni
            self.ni = ni

        try:
            # Attach to shared memory segment
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.buffer = np.ndarray(
                (self.pool_size, self.num_ao_channels, self.chunk_size),
                dtype=np.float64,
                buffer=self.shm.buf
            )

            # Initialize the chunk index counter for continuity tracking (sanity check)
            self.last_written_chunk_idx = -1  # Index of the last chunk written to the device

            # Initialize the DAQ device
            if not TEST_MODE:
                self._configure_device()

            # Preload the buffer before starting the task
            self._initialize_device(timeout=5.0)

            # Register the callback
            if not TEST_MODE:
                self.task.register_every_n_samples_transferred_from_buffer_event(
                    every_n_samples=self.chunk_size,
                    callback=self._every_chunk_samples_callback
                )
            
                # Start the task
                self.task.start()

            # Wait for the interrupt or end of stream
            self.running = True
            while self.running:
                time.sleep(0.01)
                if TEST_MODE:
                    self._every_chunk_samples_callback(None, None, None, None)

        except Exception as e:
            print(f"Writer: {e}.")
            # Report that the task is stopped (will be terminated by the manager)
            os.write(self.report_w, self.REPORT_STRUCT.pack(-1, -1))
        finally:
            self.cleanup()

    def _every_chunk_samples_callback(self, task_handle, event_type, n_samples, callback_data):
        """
        Callback function to write data to the DAQ device.
        This is called every time a chunk of samples is transferred from the buffer.
        """
        # Wait for the ready signal from the pipe up to timeout
        rlist, _, _ = select.select([self.ready_r], [], [], self.timeout)
        if not rlist:
            raise UnderrunError(f"Timed out after {self.timeout*1e3:.3f}ms waiting for ready signal from the pipe")
        
        # If we got a message from the manager, read it
        read_msg = os.read(self.ready_r, self.READY_STRUCT.size)
        if len(read_msg) < self.READY_STRUCT.size:
            raise ValueError("Received incomplete ready signal data.")
        
        # Unpack the message into chunk index and buffer index
        chunk_idx, buf_idx = self.READY_STRUCT.unpack(read_msg)

        if chunk_idx == -1:
            print("Writer received stop message.")
            self.running = False
            return

        # Make sure the chunk index is as expected
        if chunk_idx != self.last_written_chunk_idx + 1:
            raise ValueError(f"Out-of-order error: streaming chunk {chunk_idx}, expected {self.last_written_chunk_idx + 1}")
        
        # Get the flattened view of the current chunk memory
        flat_buffer = self.buffer[buf_idx].reshape(-1)

        # Send the data to the device
        try:
            if not TEST_MODE:
                self.writer.write_many_sample(flat_buffer)                                                                                                                                    
        except self.ni.errors.DaqError as e:
            print(f"DAQ error: {e}")
            raise

        # Update the last chunk index written
        self.last_written_chunk_idx = chunk_idx

        # Report back that the buffer slot is free
        os.write(self.report_w, self.REPORT_STRUCT.pack(chunk_idx, buf_idx))

    def _configure_device(self):
        # Create a DAQ task
        self.task = self.ni.Task()

        # Add analog output channels
        for ao in self.ao_channels:
            self.task.ao_channels.add_ao_voltage_chan(
                f"{self.device_name}/ao{ao.channel_name}",
                min_val=ao.min_value,
                max_val=ao.max_value
            )

        # Add digital output channels
        # TODO: Implement digital output channel configuration

        # Calculate the total output buffer size
        outbuf_size = self.chunk_size * self.outbuf_num_chunks

        # Configure the sample clock
        self.task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate,
            sample_mode=self.ni.constants.AcquisitionType.CONTINUOUS,
            samps_per_chan=outbuf_size
        )

        # Set regeneration mode
        self.task.out_stream.regen_mode = self.ni.constants.RegenerationMode.DONT_ALLOW_REGENERATION

        # Create a writer for the analog output channels
        self.writer = self.ni.stream_writers.AnalogMultiChannelWriter(
            self.task.out_stream,
            auto_start=False
        )

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

    def cleanup(self):
        """
        Clean up all the resources specific to the Writer.
        Shared memory will be closed by the manager.
        """
        try:
            # If we have a task, wait for it to finish
            if self.task:
                print("Writer: Waiting for NI-DAQ task to finish...")
                # Wait for 2x the card buffer worth of time
                time.sleep(2.0 * self.outbuf_num_chunks * self.chunk_size / self.sample_rate)
                
                if self.task:
                    self.task.stop()
                    self.task.close()

            # Close pipes
            os.close(self.ready_r)
            os.close(self.ready_w)
            os.close(self.report_r)
            os.close(self.report_w)

        except Exception as e:
            print(f"Error during cleanup: {e}")