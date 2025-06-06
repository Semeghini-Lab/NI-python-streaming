# Writer.py
# Created by: Marcin Kalinowski & Yi Zhu

import sys
import time
import struct
import functools
import numpy as np
from NICard import NICard
from multiprocessing import Process, shared_memory
import zmq

if sys.platform == "darwin":
    TEST_MODE = True
else:
    TEST_MODE = False

class UnderrunError(Exception):
    """Custom exception for underrun errors in the DAQ device."""
    pass

class Writer(Process):
    """
    Writer process: reads chunk assignments via a pipe, writes data to DAQ devices,
    and reports results back via another pipe.

    IPC is card-specific.
    """

    READY_STRUCT = struct.Struct('>q i I')  # (chunk_idx, buf_idx, card_idx)
    REPORT_STRUCT = struct.Struct('>q i I')  # (chunk_idx, buf_idx, card_idx)

    def __init__(
            self,
            writer_id: int,
            cards: list[NICard],
            card_indices: list[int],
            outbuf_num_chunks: int,  # Total number of chunks in the card buffer
            pool_size: int,  # Size of the memory pool for calculations
            ready_ports: list[int],  # Ports for receiving ready signals
            report_ports: list[int],  # Ports for sending reports
            timeout: float = 100*1e-6,  # Grace period after next chunk is expected (in seconds)
    ) -> None:
        super().__init__(name="Writer")
        self.writer_id = writer_id
        self.cards = cards
        self.card_indices = card_indices
        self.outbuf_num_chunks = outbuf_num_chunks
        self.pool_size = pool_size
        self.timeout = 0 if timeout < 1e-3 else int(timeout * 1000)

        # Store ports
        self.ready_ports = ready_ports
        self.report_ports = report_ports

        # Longest realtime chunk time (in seconds)
        self.longest_realtime_chunk_time = max([card.chunk_size/card.sample_rate for card in self.cards])

        # Make sure we have the correct number of ports
        if len(self.ready_ports) != len(self.cards):
            raise ValueError(f"Number of ready ports ({len(self.ready_ports)}) does not match number of cards ({len(self.cards)})")
        if len(self.report_ports) != len(self.cards):
            raise ValueError(f"Number of report ports ({len(self.report_ports)}) does not match number of cards ({len(self.cards)})")

        # These will be set in run()
        self.tasks = None
        self.writers = None
        self.context = None
        self.ready_sockets = []
        self.report_sockets = []
        self.buffers = []
        self.last_written_chunk_indices = None
        self.shms = None  # Store shared memory objects

    def run(self):
        if not TEST_MODE:
            import nidaqmx as ni
            from nidaqmx.stream_writers import AnalogMultiChannelWriter, DigitalMultiChannelWriter

            self.ni = ni
            self.AMCW = AnalogMultiChannelWriter
            self.DMCW = DigitalMultiChannelWriter
        try:
            # Create ZMQ context and sockets
            self.context = zmq.Context()

            # Writer receives ready signals
            self.ready_sockets = [self.context.socket(zmq.PULL) for _ in self.cards]
            # Writer sends reports
            self.report_sockets = [self.context.socket(zmq.PUSH) for _ in self.cards]

            # Set to running state
            self.running = True
            
            # Connect to ports
            for ready_socket, ready_port in zip(self.ready_sockets, self.ready_ports):
                ready_socket.connect(f"tcp://127.0.0.1:{ready_port}")
            for report_socket, report_port in zip(self.report_sockets, self.report_ports):
                report_socket.connect(f"tcp://127.0.0.1:{report_port}")

            # Attach to shared memory segments create buffers
            self.shms = []
            self.buffers = []
            for card in self.cards:
                shm = shared_memory.SharedMemory(name=card.shm_name)
                self.shms.append(shm)
                buffer = np.ndarray(
                    (self.pool_size, card.num_channels(), card.chunk_size),
                    dtype=bool if card.is_digital else np.float64,
                    buffer=shm.buf
                )
                self.buffers.append(buffer)

            # Initialize the chunk index counter for continuity tracking (sanity check)
            self.last_written_chunk_indices = [-1] * len(self.cards)

            # Initialize the DAQ device
            if not TEST_MODE:
                self.tasks = [None] * len(self.cards)
                self.writers = [None] * len(self.cards)
                for card_idx in range(len(self.cards)):
                    self._configure_device(card_idx)

            # Preload the buffer before starting the task
            for card_idx in range(len(self.cards)):
                self._initialize_device(card_idx, timeout=5.0)

            # Register the callback
            if not TEST_MODE:
                for card_idx, card in enumerate(self.cards):
                    self.tasks[card_idx].register_every_n_samples_transferred_from_buffer_event(
                        sample_interval=card.chunk_size,
                        callback_method=functools.partial(self._every_chunk_samples_callback, card_idx=card_idx),
                    )

                # Start the tasks in order that ensures trigger is armed
                for card_idx in reversed(range(len(self.cards))):
                    print(f"Worker {self.writer_id}: starting card={self.card_indices[card_idx]}")
                    self.tasks[card_idx].start()

            # Wait for the interrupt or end of stream
            self.running = True
            while self.running:
                if TEST_MODE:
                    time.sleep(self.longest_realtime_chunk_time)
                    for card_idx, card in enumerate(self.cards):
                        self._every_chunk_samples_callback(None, None, card.chunk_size, None, card_idx=card_idx)
                else:
                    time.sleep(0.01)

        except Exception as e:
            print(f"[ERROR] Writer {self.writer_id}: {e}.")
            # Report that the task is stopped (will be terminated by the manager)
            if self.running:
                for report_socket in self.report_sockets:
                    report_socket.send(self.REPORT_STRUCT.pack(-1, -1, 0))
        finally:
            self.cleanup()

    def _every_chunk_samples_callback(self, task, event_type, n_samples, callback_data, card_idx):
        """
        Callback function to write data to the DAQ device.
        This is called every time a chunk of samples is transferred from the buffer.

        card_idx: index of the calling card within the local writer scope
        """

        if not self.running or n_samples is None:
            return 0

        # Wait for the ready signal from the pipe up to timeout (in milliseconds)        
        if self.ready_sockets[card_idx].poll(timeout=self.timeout):
            # If we got a message from the manager, read it
            data = self.ready_sockets[card_idx].recv()
            chunk_idx, buf_idx, card_idx_recv = self.READY_STRUCT.unpack(data)
            #print(f"Writer {self.writer_id}: Received ready signal for chunk {chunk_idx} in slot {buf_idx} at card {card_idx_recv}.")
        else:
            raise UnderrunError(f"Timed out after {self.timeout:.2f}ms waiting for ready signal from the card {card_idx} socket")
        
        if chunk_idx == -1:
            print(f"Writer {self.writer_id} received stop message.")
            self.running = False
            return 0
        
        if card_idx_recv != self.card_indices[card_idx]:
            raise ValueError(f"Card index mismatch: {card_idx_recv} != {self.card_indices[card_idx]}")
        
        # Get the last written chunk index for the card
        last_written_chunk_idx = self.last_written_chunk_indices[card_idx]

        # Make sure the chunk index is as expected
        if chunk_idx != last_written_chunk_idx + 1:
            raise ValueError(f"Out-of-order error: streaming chunk {chunk_idx}, expected {last_written_chunk_idx + 1}")

        # Send the data to the device
        try:
            if not TEST_MODE:
                self.writers[card_idx].write_many_sample(self.buffers[card_idx][buf_idx])                                                                                                                                    
        except self.ni.errors.DaqError as e:
            print(f"DAQ error: {e}")
            raise

        # Update the last chunk index written
        self.last_written_chunk_indices[card_idx] = chunk_idx

        # Report back that the buffer slot is free
        self.report_sockets[card_idx].send(self.REPORT_STRUCT.pack(chunk_idx, buf_idx, card_idx_recv))

        return 0

    def _configure_device(self, card_idx: int):
        card = self.cards[card_idx]

        # Create DAQ tasks
        task = self.ni.Task()

        # Add channels to the task
        if card.is_digital:
            for do in card.sequences:
                task.do_channels.add_do_chan(
                    f"{card.device_name}/{do.channel_id}"
                )
        else:
            for ao in card.sequences:
                print(f"Writer {self.writer_id}: connecting to {card.device_name}/{ao.channel_id}")
                task.ao_channels.add_ao_voltage_chan(
                    f"{card.device_name}/{ao.channel_id}",
                min_val=ao.min_value,
                max_val=ao.max_value
            )

        # Calculate the total output buffer size
        outbuf_size = card.chunk_size * self.outbuf_num_chunks

        # Configure the sample clock for the card
        task.timing.cfg_samp_clk_timing(
            rate=card.sample_rate,
            sample_mode=self.ni.constants.AcquisitionType.CONTINUOUS,
            samps_per_chan=outbuf_size
        )

        # Configure the trigger for the card if it is specified
        if card.trigger_source:
            if card.is_primary:
                task.export_signals.export_signal(
                    signal_id=self.ni.constants.Signal.START_TRIGGER, 
                    output_terminal=card.trigger_source
                )
            else:
                task.triggers.start_trigger.cfg_dig_edge_start_trig(
                    card.trigger_source,
                    self.ni.constants.Edge.RISING
                )
        
        # Configure the clock source for the card
        # if card.clock_source:
        #     if card.is_primary:
        #         task.export_signals.ref_clk_output_terminal = card.clock_source
        #     else:
        #         task.timing.cfg_dig_edge_clk_src(
        #             card.clock_source
        #         )

        print(f"Writer {self.writer_id}: card={self.card_indices[card_idx]} device={card.device_name}.")

        # Set regeneration mode
        task.out_stream.regen_mode = self.ni.constants.RegenerationMode.DONT_ALLOW_REGENERATION

        # Configure the output buffer size for analog output
        task.out_stream.output_buf_size = outbuf_size


        # Create a writer for the analog output channels
        self.writers[card_idx] = (self.DMCW if card.is_digital else self.AMCW)(
            task.out_stream,
            auto_start=False
        )

        # Save the tasks
        self.tasks[card_idx] = task

    def _initialize_device(self, card_idx: int, timeout=1.0):
        """
        Load the device buffer until full.
        """

        realtime_timeout = self.timeout

        # Set longer timeout for initial buffer loading
        self.timeout = 0 if timeout < 1e-3 else int(timeout * 1000)

        # Preload the buffer
        for i in range(self.outbuf_num_chunks):
            self._every_chunk_samples_callback(None, None, self.cards[card_idx].chunk_size, None, card_idx=card_idx)

        # Reset the timeout to the original value
        self.timeout = realtime_timeout

    def cleanup(self):
        """
        Clean up all the resources specific to the Writer.
        Shared memory will be closed by the manager.
        """
        try:
            # Wait for 2x the longest card buffer time
            print(f"Writer {self.writer_id}: Waiting for NI-DAQ task to finish...")
            time.sleep(2.0 * self.longest_realtime_chunk_time * self.outbuf_num_chunks)

            # If we have a task, wait for it to finish
            for task in self.tasks:
                if task:
                    task.stop()
                    task.close()

            # Close ZMQ sockets and context
            for ready_socket in self.ready_sockets:
                ready_socket.close()
            for report_socket in self.report_sockets:
                report_socket.close()

        except Exception as e:
            print(f"Writer {self.writer_id}: Error during cleanup: {e}")