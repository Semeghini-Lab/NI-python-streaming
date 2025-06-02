# Writer.py
# Created by: Marcin Kalinowski & Yi Zhu

import os
import time
import struct
import select
import numpy as np
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

    READY_STRUCT = struct.Struct('>q i')  # (seq, buf_idx)
    REPORT_STRUCT = struct.Struct('>i')  # (buf_idx)

    def __init__(
            self,
            shm_name: str,  # Name of the shared memory segment
            sample_rate: float,  # Sample rate in Hz
            num_channels: int,  # Total number of channels
            chunk_size: int,  # Size of each chunk in samples
            outbuf_num_chunks: int,  # Total number of chunks in the card buffer
            pool_size: int,  # Size of the memory pool for calculations
            device_name: str,  # Name of the DAQ device
            ao_channels: list,  # List of analog output channel named tuples
            do_channels: list,  # List of digital output channels named tuples
            timeout: float = 100*1e-6,  # Grace period after next chunk is expected
    ) -> None:
        super().__init__(name="Writer")
        self.shm_name = shm_name
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.outbuf_num_chunks = outbuf_num_chunks
        self.pool_size = pool_size
        self.num_channels = num_channels
        self.device_name = device_name
        self.ao_channels = ao_channels
        self.do_channels = do_channels
        self.timeout = timeout

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
                (self.pool_size, self.num_channels, self.chunk_size),
                dtype=np.float64,
                buffer=self.shm.buf
            )

            # Initialize the sequence counter for continuity tracking (sanity check)
            self.last_seq_written = -1  # Last sequence number written to the device

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

            # Wait for the interrupt
            while True:
                time.sleep(0.1)
                if TEST_MODE:
                    self._every_chunk_samples_callback(None, None, None, None)

        except KeyboardInterrupt:
            print("Writer: Keyboard interrupt received.")
        except Exception as e:
            print(f"Writer failed: {e}.")
        finally:
            # Cleanup
            if self.task:
                print("Writer: Stopping the NI-DAQ task.")
                self.task.stop()
                self.task.close()
            if self.shm:
                self.shm.close()

            # Report that the task is stopped
            os.write(self.report_w, self.REPORT_STRUCT.pack(-1))

    def _every_chunk_samples_callback(self, task_handle, event_type, n_samples, callback_data):
        """
        Callback function to write data to the DAQ device.
        This is called every time a chunk of samples is transferred from the buffer.
        """
        # Wait for the ready signal from the pipe up to timeout
        rlist, _, _ = select.select([self.ready_r], [], [], self.timeout)
        if not rlist:
            raise UnderrunError(f"Timed out after {self.timeout*1e3:.3f}ms waiting for ready signal from the pipe")
        
        # If we got a signal, read it
        read_msg = os.read(self.ready_r, self.READY_STRUCT.size)
        if len(read_msg) < self.READY_STRUCT.size:
            raise ValueError("Received incomplete ready signal data.")
        
        # Unpack the ready signal
        seq, buf_idx = self.READY_STRUCT.unpack(read_msg)

        # Make sure the sequence number is as expected
        if seq != self.last_seq_written + 1:
            raise ValueError(f"Out-of-order error: streaming chunk {seq}, expected {self.last_seq_written + 1}")
        
        # Get the flattened view of the current chunk memory
        flat_buffer = self.buffer[buf_idx].reshape(-1)

        # Send the data to the device
        try:
            if not TEST_MODE:
                self.writer.write_many_sample(flat_buffer)                                                                                                                                    
        except self.ni.errors.DaqError as e:
            print(f"DAQ error: {e}")
            raise

        # Update the last sequence number written
        self.last_seq_written = seq

        # Report back that the buffer is free
        os.write(self.report_w, self.REPORT_STRUCT.pack(buf_idx))

    def _configure_device(self):
        self.task = self.ni.Task()
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