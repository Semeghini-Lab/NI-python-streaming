# Writer.py
# Created by: Marcin Kalinowski & Yi Zhu

import sys
import time
import struct
import numpy as np
from multiprocessing import Process, shared_memory
import zmq
from nidaqmx.constants import LineGrouping

from nistreamer.NICard import NICard

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
            num_all_cards: int,  # Total number of cards in the system
            ready_ports: list[int],  # Ports for receiving ready signals
            report_ports: list[int],  # Ports for sending reports
            timeout: float = 0.0,  # Grace period after next chunk is expected (in seconds)
    ) -> None:
        super().__init__(name="Writer")
        self.writer_id = writer_id
        self.cards = cards
        self.card_indices = card_indices
        self.outbuf_num_chunks = outbuf_num_chunks
        self.pool_size = pool_size
        self.timeout = int(timeout)
        self.num_all_cards = num_all_cards

        # Store ports
        self.ready_ports = ready_ports
        self.report_ports = report_ports

        # How many chunks need to be written for each card
        self.num_chunks_to_write = [card.num_chunks for card in self.cards]

        # Realtime chunk time (in seconds)
        self.longest_realtime_chunk_time = max([card.chunk_size/card.sample_rate for card in self.cards])
        self.shortest_realtime_chunk_time = min([card.chunk_size/card.sample_rate for card in self.cards])

        # Check if there is a primary card
        self.has_primary_card = any([card.is_primary for card in self.cards])

        # Make sure we have the correct number of ports
        if len(self.ready_ports) != len(self.cards):
            raise ValueError(f"Number of ready ports ({len(self.ready_ports)}) does not match number of cards ({len(self.cards)})")
        if len(self.report_ports) != len(self.cards):
            raise ValueError(f"Number of report ports ({len(self.report_ports)}) does not match number of cards ({len(self.cards)})")

        # These will be set in run()
        self.tasks = None
        self.writers = None
        self.context = None
        self.active_cards = None
        self.ready_sockets = None
        self.report_sockets = None
        self.buffers = None
        self.last_written_chunk_indices = None
        self.shms = None  # Store shared memory objects

        # Shared memory for the card initialization flags
        self.shm_init_card_flags = None
        self.init_card_flags = None

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
            self.ready_sockets = [self.context.socket(zmq.PULL) for _ in self.cards]
            # Writer sends reports
            self.report_sockets = [self.context.socket(zmq.PUSH) for _ in self.cards]

            # Set to running state
            self.running = True

            # Set all cards to active
            self.active_cards = [card_idx for card_idx in range(len(self.cards))]
            
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

            # Initialize the card initialization flags buffer
            self.shm_init_card_flags = shared_memory.SharedMemory(name="nistreamer_init_card_flags")
            self.init_card_flags = np.ndarray(
                (self.num_all_cards,),
                dtype=bool,
                buffer=self.shm_init_card_flags.buf
            )

            # Initialize the chunk index counter for continuity tracking (sanity check)
            self.last_written_chunk_indices = [-1] * len(self.cards)

            # Initialize the DAQ device
            if not TEST_MODE:
                self.tasks = [None] * len(self.cards)
                self.writers = [None] * len(self.cards)
                for card_idx in self.active_cards:
                    self._configure_device(card_idx)

            # Preload the buffer before starting the task
            for card_idx in self.active_cards:
                self._initialize_device(card_idx, timeout=5.0)

            # If there is a primary card, wait until all cards are initialized
            if self.has_primary_card:
                while not all(self.init_card_flags):
                    time.sleep(0.00001) # Wait for 10 microseconds

            # Start the tasks in order that ensures trigger is armed
            if not TEST_MODE:
                for card_idx in reversed(self.active_cards):
                    print(f"Writer {self.writer_id}: starting card={self.card_indices[card_idx]}.")
                    self.tasks[card_idx].start()

            # Start polling loop instead of using callbacks
            self.running = True
            self._polling_loop()

        except Exception as e:
            print(f"[ERROR] Writer {self.writer_id}: {e}.")
            # Report that the task is stopped (will be terminated by the manager)
            if self.running:
                for report_socket in self.report_sockets:
                    report_socket.send(self.REPORT_STRUCT.pack(-1, -1, 0))
        finally:
            self.cleanup()

    def _polling_loop(self):
        """
        Main polling loop that checks buffer space availability and writes data when ready.
        This replaces the callback-based approach.
        """
        while self.running:
            for card_idx in self.active_cards:
                # Check if there's enough space in the buffer for a chunk
                if not TEST_MODE:
                    card = self.cards[card_idx]
                    
                    # Debug: catch underrun digital cards that are not reporting
                    if self.tasks[card_idx].out_stream.space_avail == 0:
                        raise ValueError(f"BROKEN CARD at card={self.card_indices[card_idx]}")
                    
                    # Keep filling the buffer while there's space - be more aggressive
                    while self.tasks[card_idx].out_stream.space_avail >= card.chunk_size and self.running:
                        if not self._write_chunk_to_device(card_idx):
                            # No more data available from manager, break inner loop
                            break
                else:
                    # In test mode, simulate writing at the expected rate
                    time.sleep(self.shortest_realtime_chunk_time)
                    if not self._write_chunk_to_device(card_idx):
                        # Broken in the testing mode, raise an error
                        raise ValueError(f"chunk={self.last_written_chunk_indices[card_idx]+1} was not ready at card={self.card_indices[card_idx]}")

    def _write_chunk_to_device(self, card_idx):
        """
        Write a chunk of data to the specified device.
        This is equivalent to the old callback function but called from polling loop.
        
        card_idx: index of the card within the local writer scope
        Returns: True if data was written, False if no data available
        """
        if not self.running:
            return False

        # Wait for the ready signal from the pipe up to timeout (in milliseconds)        
        if self.ready_sockets[card_idx].poll(timeout=self.timeout):
            # If we got a message from the manager, read it
            data = self.ready_sockets[card_idx].recv()
            chunk_idx, buf_idx, card_idx_recv = self.READY_STRUCT.unpack(data)
            #print(f"Writer {self.writer_id}: Received ready signal for chunk {chunk_idx} in slot {buf_idx} at card {card_idx_recv}.")
        else:
            return False
        
        if chunk_idx == -1:
            print(f"Writer {self.writer_id} received stop message.")
            self.running = False
            # Immediately stop DAQ tasks to prevent underrun errors during shutdown
            if not TEST_MODE and self.tasks:
                print(f"Writer {self.writer_id}: Stopping DAQ tasks immediately to prevent underrun...")
                for task_idx, task in enumerate(self.tasks):
                    if task:
                        try:
                            task.stop()
                            print(f"Writer {self.writer_id}: Stopped DAQ task {task_idx}")
                        except Exception as e:
                            print(f"Writer {self.writer_id}: Error stopping task {task_idx}: {e}")
            return False
        
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
                card = self.cards[card_idx]
                if card.is_digital:
                    digital_data = self.buffers[card_idx][buf_idx]
                    # For single digital channel, convert from (1, samples) to (samples,)
                    if digital_data.shape[0] == 1:
                       digital_data = digital_data.squeeze(axis=0)  # Remove the first dimension
                    self.tasks[card_idx].write(digital_data)
                else:
                    # Use standard write method for analog cards
                    self.writers[card_idx].write_many_sample(self.buffers[card_idx][buf_idx])
        except self.ni.errors.DaqError as e:
            print(f"DAQ error: {e}")
            raise

        # Update the last chunk index written
        self.last_written_chunk_indices[card_idx] = chunk_idx

        # Report back that the buffer slot is free
        self.report_sockets[card_idx].send(self.REPORT_STRUCT.pack(chunk_idx, buf_idx, card_idx_recv))

        # If the last chunk was written, remove the card from the active cards
        if chunk_idx == self.num_chunks_to_write[card_idx]-1:
            print(f"Writer {self.writer_id}: finished writing card={self.card_indices[card_idx]}.")
            self.active_cards.remove(card_idx)
        
        # Return True to indicate successful write
        return True

    def _configure_device(self, card_idx: int):
        card = self.cards[card_idx]

        # Create DAQ tasks
        task = self.ni.Task()

        # Add channels to the task
        if card.is_digital:
            for do in card.sequences:
                task.do_channels.add_do_chan(
                    f"{card.device_name}/{do.channel_id}",
                    line_grouping=LineGrouping.CHAN_PER_LINE
                )
        else:
            for ao in card.sequences:
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
        if card.clock_source:
            if card.is_primary:
                task.export_signals.export_signal(
                    signal_id=self.ni.constants.Signal.TEN_MHZ_REF_CLOCK,
                    output_terminal=card.clock_source
                )
            else:
                if card.is_digital:
                    task.timing.samp_clk_rate = 10e6
                    task.timing.samp_clk_src = card.clock_source
                else:
                    task.timing.ref_clk_rate = 10e6
                    task.timing.ref_clk_src = card.clock_source

        print(f"Writer {self.writer_id}: card={self.card_indices[card_idx]} {'(primary)' if card.is_primary else ''} device={card.device_name} trigger={card.trigger_source} clock={card.clock_source}.")

        # Set regeneration mode
        task.out_stream.regen_mode = self.ni.constants.RegenerationMode.DONT_ALLOW_REGENERATION

        # Configure the output buffer size for analog output
        task.out_stream.output_buf_size = outbuf_size

        # Create a writer for analog output channels only
        if card.is_digital:
            self.writers[card_idx] = None
        else:
            self.writers[card_idx] = self.AMCW(
                task.out_stream,
                auto_start=False
            )

        # Save the tasks
        self.tasks[card_idx] = task

    def _initialize_device(self, card_idx: int, timeout=5.0):
        """
        Load the device buffer until full.
        """
        realtime_timeout = self.timeout

        # Set longer timeout for initial buffer loading (5 seconds default)
        self.timeout = 0 if timeout < 1e-3 else int(timeout * 1000) # Convert to milliseconds

        print(f"Writer {self.writer_id}: starting to fill buffer of {self.outbuf_num_chunks} chunks for card={self.card_indices[card_idx]}.")
        
        # Preload the buffer
        chunks_loaded = 0
        for i in range(self.outbuf_num_chunks):
            if self._write_chunk_to_device(card_idx):
                chunks_loaded += 1
            else:
                print(f"Writer {self.writer_id}: warning - only loaded {chunks_loaded}/{self.outbuf_num_chunks} chunks during initialization")
                break
        
        print(f"Writer {self.writer_id}: successfully preloaded {chunks_loaded} chunks for card={self.card_indices[card_idx]}.")

        # Mark the card as initialized
        self.init_card_flags[self.card_indices[card_idx]] = True

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
            time.sleep(1.2 * self.longest_realtime_chunk_time * self.outbuf_num_chunks)

            # If we have a task, wait for it to finish
            if self.tasks:
                for task_idx, task in enumerate(self.tasks):
                    if task:
                        try:
                            # Stop task if not already stopped
                            if not task.is_task_done():
                                print(f"Writer {self.writer_id}: Task {task_idx} still running, stopping...")
                                task.stop()
                            else:
                                print(f"Writer {self.writer_id}: Task {task_idx} already stopped")
                        except Exception as e:
                            print(f"Writer {self.writer_id}: Error stopping task {task_idx}: {e}")
                        try:
                            task.close()
                            print(f"Writer {self.writer_id}: Closed task {task_idx}")
                        except Exception as e:
                            print(f"Writer {self.writer_id}: Error closing task {task_idx}: {e}")

            # Close ZMQ sockets and context
            for ready_socket in self.ready_sockets:
                ready_socket.close()
            for report_socket in self.report_sockets:
                report_socket.close()

        except Exception as e:
            print(f"Writer {self.writer_id}: Error during cleanup: {e}")