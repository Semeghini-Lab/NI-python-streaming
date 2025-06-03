# Writer.py
# Created by: Marcin Kalinowski & Yi Zhu

import os
import time
import struct
import numpy as np
from AOSequence import AOSequence
from multiprocessing import Process, shared_memory
import zmq

TEST_MODE = False

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
            ready_port: int,  # Port for receiving ready signals
            report_port: int,  # Port for sending reports
            timeout: float = 100*1e-6,  # Grace period after next chunk is expected (in seconds)
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
        self.timeout = 0 if timeout < 1e-3 else int(timeout * 1000)

        # Store ports
        self.ready_port = ready_port
        self.report_port = report_port

        # These will be set in run()
        self.shm = None
        self.buffer = None
        self.task = None
        self.writer = None
        self.context = None
        self.ready_socket = None
        self.report_socket = None

    def run(self):
        if not TEST_MODE:
            import nidaqmx as ni
            from nidaqmx.stream_writers import AnalogMultiChannelWriter

            self.ni = ni
            self.AMCW = AnalogMultiChannelWriter

        try:
            # Create ZMQ context and sockets
            self.context = zmq.Context()
            # Writer receives ready signals
            self.ready_socket = self.context.socket(zmq.PULL)
            # Writer sends reports
            self.report_socket = self.context.socket(zmq.PUSH)

            # Set to running state
            self.running = True
            
            # Connect to ports
            self.ready_socket.connect(f"tcp://127.0.0.1:{self.ready_port}")
            self.report_socket.connect(f"tcp://127.0.0.1:{self.report_port}")
            
            print(f"Writer: Connected to ports ready={self.ready_port}, report={self.report_port}")

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
                    self.chunk_size,
                    self._every_chunk_samples_callback
                )
            
                # Start the task
                self.task.start()

            # Wait for the interrupt or end of stream
            self.running = True
            while self.running:
                time.sleep(0.01)
                #if TEST_MODE:
                #    self._every_chunk_samples_callback(None, None, None, None)

        except Exception as e:
            print(f"Writer: {e}.")
            # Report that the task is stopped (will be terminated by the manager)
            if self.running:
                self.report_socket.send(self.REPORT_STRUCT.pack(-1, -1))
        finally:
            self.cleanup()

    def _every_chunk_samples_callback(self, task_handle, event_type, n_samples, callback_data):
        """
        Callback function to write data to the DAQ device.
        This is called every time a chunk of samples is transferred from the buffer.
        """
        if not self.running or n_samples is None:
            return 0

        # Wait for the ready signal from the pipe up to timeout (in milliseconds)        
        if self.ready_socket.poll(timeout=self.timeout):
            # If we got a message from the manager, read it
            data = self.ready_socket.recv()
            chunk_idx, buf_idx = self.READY_STRUCT.unpack(data)
        else:
            raise UnderrunError(f"Timed out after {self.timeout:.2f}ms waiting for ready signal from the pipe")
        
        if chunk_idx == -1:
            print("Writer received stop message.")
            self.running = False
            return 0

        # Make sure the chunk index is as expected
        if chunk_idx != self.last_written_chunk_idx + 1:
            raise ValueError(f"Out-of-order error: streaming chunk {chunk_idx}, expected {self.last_written_chunk_idx + 1}")

        # Send the data to the device
        try:
            if not TEST_MODE:
                self.writer.write_many_sample(self.buffer[buf_idx])                                                                                                                                    
        except self.ni.errors.DaqError as e:
            print(f"DAQ error: {e}")
            raise

        # Update the last chunk index written
        self.last_written_chunk_idx = chunk_idx

        # Report back that the buffer slot is free
        self.report_socket.send(self.REPORT_STRUCT.pack(chunk_idx, buf_idx))

        return 0

    def _configure_device(self):
        # Create a DAQ task
        self.task = self.ni.Task()

        # Add analog output channels
        for ao in self.ao_channels:
            self.task.ao_channels.add_ao_voltage_chan(
                f"{self.device_name}/{ao.channel_name}",
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

        # Configure the output buffer size
        self.task.out_stream.output_buf_size = outbuf_size

        # Create a writer for the analog output channels
        self.writer = self.AMCW(
            self.task.out_stream,
            auto_start=False
        )

    def _initialize_device(self, timeout=1.0):
        """Load the device buffer until full."""
        realtime_timeout = self.timeout

        # Set longer timeout for initial buffer loading
        self.timeout = 0 if timeout < 1e-3 else int(timeout * 1000)

        # Preload the buffer
        for i in range(self.outbuf_num_chunks):
            print(f"Writer: Preloading chunk {i + 1}/{self.outbuf_num_chunks}...")
            self._every_chunk_samples_callback(-1, -1, -1, -1)

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

            # Close ZMQ sockets and context
            self.ready_socket.close()
            self.report_socket.close()
            self.context.term()

        except Exception as e:
            print(f"Error during cleanup: {e}")