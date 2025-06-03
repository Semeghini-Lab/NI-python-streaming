# SequenceStreamer.py
# Created by: Marcin Kalinowski & Yi Zhu

import os, sys
import time
import struct
import numpy as np
import selectors
import zmq

from AOSequence import AOSequence

from Worker import Worker
from Writer import Writer

import multiprocessing

if sys.platform == "darwin":
    multiprocessing.set_start_method("fork")


class SequenceStreamer:
    """
    Manager class: sets up shared memory, manages worker and writer processes,
    and runs the even loop in the main thread to assign chunks and collect reports.
    """

    WORKER_ASSIGN_STRUCT = struct.Struct('>q i I I')  # (chunk_idx, buf_idx, ch_start, ch_end)
    WORKER_DONE_STRUCT = struct.Struct('>q i I d')  # (chunk_idx, buf_idx, worker_id, compute_time)
    CHUNK_READY_STRUCT = struct.Struct('>q i')  # (chunk_idx, buf_idx)
    SLOT_FREE_STRUCT = struct.Struct('>q i')  # (chunk_idx, buf_idx)

    def __init__(
            self,
            ao_seqs: list[AOSequence],  # List of analog output sequences
            chunk_size: int,  # Size of each chunk in samples
            pool_size: int,  # Size of the memory pool
            sample_rate: float,  # Sample rate in Hz
            num_workers: int,  # Number of worker processes
            device_name: str,  # Name of the DAQ device
    ) -> None:
        # TODO: add digital output support
        self.ao_seqs = ao_seqs
        self.num_channels = len(ao_seqs)
        self.chunk_size = chunk_size
        self.pool_size = pool_size
        self.sample_rate = sample_rate
        self.num_workers = min(num_workers, self.num_channels)
        self.device_name = device_name

        self.num_chunks_to_stream = int(self.ao_seqs[0].instructions[-1].end_sample // self.chunk_size)

        # Make sure all sequences have the same sample rate and stop sample
        for seq in self.ao_seqs:
            if seq.sample_rate != self.sample_rate:
                raise ValueError(f"Sequence {seq.channel_name} has sample rate {seq.sample_rate} but the sample rate is {self.sample_rate}.")
            if seq.instructions[-1].end_sample != self.num_chunks_to_stream * self.chunk_size:
                raise ValueError(f"Sequence {seq.channel_name} has stop sample {seq.instructions[-1].end_sample} but the stop sample is {self.num_chunks_to_stream * self.chunk_size}.")
            if not seq.is_compiled:
                raise ValueError(f"Sequence {seq.channel_name} is not compiled.")

        # Create shared memory segment for the data buffer
        shm_size = self.pool_size * self.num_channels * self.chunk_size * np.dtype(np.float64).itemsize
        shm_name = f"nistreamer_shared_memory_{os.getpid()}"

        # Create shared memory segment
        self.shm = multiprocessing.shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)
        self.buffer = np.ndarray(
            (self.pool_size, self.num_channels, self.chunk_size), 
            dtype=np.float64, 
            buffer=self.shm.buf
        )
        self.buffer.fill(0.0)

        print(f"Shared memory segment '{shm_name}' created with size {shm_size} bytes: ({self.pool_size}x [{self.chunk_size} chunk size for {self.num_channels} channels]).")

        # Initialize ZMQ context and sockets for writer communication
        self.context = zmq.Context()

        self.ready_socket = None
        self.report_socket = None

        # Initialize ZMQ sockets for worker communication
        self.worker_assign_sockets = []  # PUSH sockets for sending assignments
        self.worker_done_sockets = []    # PULL sockets for receiving completion reports

        # Initialize a callback map for monitored events
        self.event_callbacks = {}

        # Create poller for dispatching messages
        self.poller = zmq.Poller()

        # Compute channel ranges
        channels_per_worker = self.num_channels // self.num_workers
        self.channel_ranges = []
        for i in range(self.num_workers-1):
            self.channel_ranges.append((i*channels_per_worker, (i+1)*channels_per_worker))
        # The last worker gets the remaining channels
        self.channel_ranges.append(((self.num_workers-1)*channels_per_worker, self.num_channels))

        # Placeholder for processes
        self.workers = []
        self.writer = None

        # Manager internal state
        self.next_calc_chunk = 0
        self.next_calc_channel_group = 0
        self.available_workers = []
        self.available_slots = [i for i in range(self.pool_size)]  # List of available buffer slots
        self.processed_chunks = {} # Dictionary of of the form {chunk: set(remaining channel groups)]}
        self.processed_chunk_slots = {} # Keep track of which chunk is processed in which slots (sanity check)

    def start(self):
        try:
            # Spawn worker processes
            for wid in range(self.num_workers):
                # Create and bind worker assign socket
                assign_socket = self.context.socket(zmq.PUSH)
                assign_port = assign_socket.bind_to_random_port("tcp://127.0.0.1")
                self.worker_assign_sockets.append(assign_socket)
                print(f"SequenceStreamer: Bound worker {wid} assign socket to port {assign_port}")

                # Create and bind worker done socket
                done_socket = self.context.socket(zmq.PULL)
                done_port = done_socket.bind_to_random_port("tcp://127.0.0.1")
                self.worker_done_sockets.append(done_socket)
                print(f"SequenceStreamer: Bound worker {wid} done socket to port {done_port}")

                # Register worker done socket with poller
                self.poller.register(done_socket, zmq.POLLIN)
                self.event_callbacks[done_socket] = self._process_worker_done_data

                worker = Worker(
                    worker_id=wid,
                    ao_channels=self.ao_seqs,
                    shm_name=self.shm.name,
                    sample_rate=self.sample_rate,
                    chunk_size=self.chunk_size,
                    num_channels=self.num_channels,
                    pool_size=self.pool_size,
                    assign_port=assign_port,
                    done_port=done_port,
                )
                worker.daemon = True
                worker.start()

                # Add as available worker
                self.available_workers.append(wid)

            # Create and bind writer sockets
            self.ready_socket = self.context.socket(zmq.PUSH)
            self.report_socket = self.context.socket(zmq.PULL)
            
            # Bind ZMQ sockets to random ports
            self.ready_port = self.ready_socket.bind_to_random_port("tcp://127.0.0.1")
            self.report_port = self.report_socket.bind_to_random_port("tcp://127.0.0.1")
            print(f"SequenceStreamer: Bound writer sockets to ports ready={self.ready_port}, report={self.report_port}")

            # Register writer sockets with poller
            self.poller.register(self.report_socket, zmq.POLLIN)
            self.event_callbacks[self.report_socket] = self._process_slot_free

            # Spawn writer process
            self.writer = Writer(
                shm_name=self.shm.name,
                sample_rate=self.sample_rate,
                chunk_size=self.chunk_size,
                outbuf_num_chunks=self.num_chunks_to_stream,
                pool_size=self.pool_size,
                device_name=self.device_name,
                ao_channels=self.ao_seqs,
                do_channels=[],
                ready_port=self.ready_port,
                report_port=self.report_port,
            )
            self.writer.daemon = True
            self.writer.start()

            # Start the manager loop
            self._manager_loop()

        except Exception as e:
            print(f"Error in start: {e}")
            self.cleanup()
            raise

    def _assign_chunk(self, worker_id: int, chunk_idx: int, buf_idx: int, ch_start: int, ch_end: int):
        """Assign a chunk to a worker."""
        assign_data = self.WORKER_ASSIGN_STRUCT.pack(chunk_idx, buf_idx, ch_start, ch_end)
        print(f"Assigning chunk {chunk_idx} to worker {worker_id}")
        self.worker_assign_sockets[worker_id].send(assign_data)

    def _process_worker_done_data(self, socket):
        """Process worker done data."""
        data = socket.recv()
        chunk_idx, buf_idx, worker_id, compute_time = self.WORKER_DONE_STRUCT.unpack(data)

        if chunk_idx not in self.processed_chunks:
            raise ValueError(f"Received done notification for unknown chunk {chunk_idx}.")
        
        if chunk_idx not in self.processed_chunk_slots:
            raise ValueError(f"Received done notification for unknown chunk {chunk_idx} in slots.")
        
        if self.processed_chunk_slots[chunk_idx] != buf_idx:
            raise ValueError(f"Received done notification for chunk {chunk_idx} in unexpected slot {buf_idx} vs {self.processed_chunk_slots[chunk_idx]}.")

        # Free the worker
        self.available_workers.append(worker_id)

        # Decrement the number of processed channel groups for this chunk
        self.processed_chunks[chunk_idx] -= 1

        # All channel groups for this chunk have been processed
        if self.processed_chunks[chunk_idx] == 0:
            # Remove the chunk from processed chunks
            del self.processed_chunks[chunk_idx]
            del self.processed_chunk_slots[chunk_idx]

            # Write the chunk ready notification to the writer
            self._assign_slot_write(chunk_idx, buf_idx)

    def _process_slot_free(self, socket):
        """
        Process a slot free notification from the writer.
        This method reads the slot index from the pipe and adds it as available.
        """
        data = socket.recv()
        chunk_idx, buf_idx = self.SLOT_FREE_STRUCT.unpack(data)
        
        # Add the slot to the available slot list
        self.available_slots.append(buf_idx)

        # If it was the last chunk, stop the manager loop
        if chunk_idx == self.num_chunks_to_stream - 1:
            self.running = False
            print("Last chunk streamed, stopping manager loop.")

    def _assign_slot_write(self, chunk_idx: int, buf_idx: int):
        """
        Notify the writer that a chunk is ready to be written.
        This method writes the chunk index and buffer index to the pipe.
        """
        print(f"Sending ready signal for chunk {chunk_idx} in slot {buf_idx}")
        self.ready_socket.send(self.CHUNK_READY_STRUCT.pack(chunk_idx, buf_idx))

    def _manager_loop(self):
        print(f"Starting manager loop with {self.num_chunks_to_stream} chunks to stream.")
        # Start the manager loop
        self.running = True

        while self.running:
            # While there are free workers and available slots, assign chunks to be computed until done
            while self.next_calc_chunk < self.num_chunks_to_stream and len(self.available_workers):
                # If the chunk is already processed, get the slot it is in
                if self.next_calc_chunk in self.processed_chunks:
                    buf_idx = self.processed_chunk_slots[self.next_calc_chunk]
                else:
                    # If the chunk is not processed, get the next available slot
                    if len(self.available_slots):
                        buf_idx = self.available_slots.pop()
                        self.processed_chunk_slots[self.next_calc_chunk] = buf_idx
                        self.processed_chunks[self.next_calc_chunk] = self.num_workers
                    else:
                        break

                # Get the next available worker
                worker_id = self.available_workers.pop()

                # Determine the channel range for this worker
                ch_start, ch_end = self.channel_ranges[self.next_calc_channel_group]

                # Assign the chunk to the worker
                self._assign_chunk(worker_id, self.next_calc_chunk, buf_idx, ch_start, ch_end)

                # Increment the sequence number if necessary
                self.next_calc_channel_group += 1
                if self.next_calc_channel_group >= self.num_workers:
                    self.next_calc_channel_group = 0
                    self.next_calc_chunk += 1

            # Poll for messages from workers and writer
           # socks = dict()
            
            # Process events
            for socket, _ in self.poller.poll(0):
                self.event_callbacks[socket](socket)

        # Send messages to stop the writer and workers
        for socket in self.worker_assign_sockets:
            socket.send(self.WORKER_ASSIGN_STRUCT.pack(-1, -1, 0, 0))

        self.ready_socket.send(self.CHUNK_READY_STRUCT.pack(-1, -1))

        # Wait for the writer and workers to finish
        self.writer.join(timeout=1.0) # in seconds
        for worker in self.workers:
            worker.join(timeout=0.1) # in seconds

        print("Sequence completed.")

    def cleanup(self):
        """Clean up all resources."""
        try:
            try:
                self.shm.close()
                self.shm.unlink()
            except FileNotFoundError:
                pass

            # Close ZMQ sockets and context
            if self.ready_socket:
                self.ready_socket.close()
            if self.report_socket:
                self.report_socket.close()
            for socket in self.worker_assign_sockets:
                socket.close()
            for socket in self.worker_done_sockets:
                socket.close()
            self.context.term()

        except Exception as e:
            print(f"Error during cleanup: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        self.cleanup()


        
if __name__ == "__main__":

    # Get the sample rate
    sample_rate = 1e6

    # Channel 0
    ch0 = AOSequence(channel_name="ao0", sample_rate=sample_rate)
    ch0.const(1.0, 1.0, value=5.0)
    ch0.linramp(3.0, 1.0, start=0, end=10)
    ch0.sine(6.0, 0.75, freq=1, amp=2, phase=0)

    # Channel 1
    ch1 = AOSequence(channel_name="ao2", sample_rate=sample_rate)
    ch1.const(1.0, 1.0, value=5.0)
    ch1.linramp(3.0, 1.0, start=0, end=10)
    ch1.sine(6.0, 0.75, freq=1, amp=2, phase=0)

    # Aggregate
    ao_seqs = [ch0, ch1]

    # Set the chunk size and compile 
    chunk_size = 65536

    # Find the maximum stop sample
    max_sample_time = np.max([seq.instructions[-1].end_sample for seq in ao_seqs])

    # Complete to a full number of chunks
    max_stop_sample = np.ceil(max_sample_time / chunk_size) * chunk_size

    # Compile
    for seq in ao_seqs:
        seq.compile(stopsamp=max_stop_sample)

    # Example usage with context manager
    with SequenceStreamer(
        ao_seqs=ao_seqs,
        chunk_size=chunk_size,  # Size of each chunk in samples
        pool_size=8,  # Size of the memory pool
        sample_rate=sample_rate,  # Sample rate in Hz
        num_workers=4,  # Number of worker processes
        device_name="PXI1Slot3"  # Name of the DAQ device
    ) as streamer:
        streamer.start()

