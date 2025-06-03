# SequenceStreamer.py
# Created by: Marcin Kalinowski & Yi Zhu

import os
import time
import struct
import numpy as np
import selectors

from AOSequence import AOSequence

from Worker import Worker
from Writer import Writer, TEST_MODE

import multiprocessing
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

        # Initialize file descriptors for inter-process communication with the writer.
        self.chunk_ready_child_fd, self.chunk_ready_fd = None, None
        self.slot_free_fd, self.slot_free_child_fd = None, None

        # Initialize file descriptors for inter-process communication with the workers.
        self.worker_assign_fds = []
        self.worker_done_fds = []

        self.worker_assign_child_fds = []
        self.worker_done_child_fds = []

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

        # Selector for worker and free pipes
        self.sel = selectors.DefaultSelector()

    def _process_worker_done(self, fd):
        """
        Process a worker done notification.
        This method reads the sequence number and buffer index from the pipe,
        marks the slot as available, and updates the processed sequences.
        """
        data = os.read(fd, self.WORKER_DONE_STRUCT.size)
        if len(data) < self.WORKER_DONE_STRUCT.size:
            raise ValueError("Received incomplete worker done notification.")
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

    def _process_slot_free(self, fd):
        """
        Process a slot free notification from the writer.
        This method reads the slot index from the pipe and adds it as available.
        """
        data = os.read(fd, self.SLOT_FREE_STRUCT.size)
        if len(data) < self.SLOT_FREE_STRUCT.size:
            raise ValueError("Received incomplete slot free notification.")
        
        chunk_idx, buf_idx = self.SLOT_FREE_STRUCT.unpack(data)
        
        # Add the slot to the available slot list
        self.available_slots.append(buf_idx)

        # If it was the last chunk, stop the manager loop
        if chunk_idx == self.num_chunks_to_stream - 1:
            self.running = False
            print("Last chunk streamed, stopping manager loop.")

    def _assign_chunk(self, worker_id: int, chunk_idx: int, buf_idx: int, ch_start: int, ch_end: int):
        """Assign a chunk to a worker."""
        assign_data = self.WORKER_ASSIGN_STRUCT.pack(chunk_idx, buf_idx, ch_start, ch_end)
        os.write(self.worker_assign_fds[worker_id], assign_data)

    def _assign_slot_write(self, chunk_idx: int, buf_idx: int):
        """
        Notify the writer that a chunk is ready to be written.
        This method writes the chunk index and buffer index to the pipe.
        """
        ready_data = self.CHUNK_READY_STRUCT.pack(chunk_idx, buf_idx)
        os.write(self.chunk_ready_fd, ready_data)

    def _manager_loop(self):
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

            # Check for completed chunks from workers and free slots from writer
            events = self.sel.select(timeout=1e-6)
            for key, _ in events:
                callback = key.data
                callback(key.fileobj)

        # Send messages to stop the writer and workers
        for fd in self.worker_assign_fds:
            os.write(fd, self.WORKER_ASSIGN_STRUCT.pack(-1, -1, 0, 0))

        os.write(self.chunk_ready_fd, self.CHUNK_READY_STRUCT.pack(-1, -1))

        # Wait for the writer and workers to finish
        self.writer.join(timeout=1.0)
        for worker in self.workers:
            worker.join(timeout=0.1)

        print("Sequence completed.")

    def start(self):
        try:
            # Spawn worker processes
            for wid in range(self.num_workers):
                worker = Worker(
                    worker_id=wid,
                    ao_channels=self.ao_seqs,
                    shm_name=self.shm.name,
                    sample_rate=self.sample_rate,
                    chunk_size=self.chunk_size,
                    num_channels=self.num_channels,
                    pool_size=self.pool_size
                )
                worker.daemon = True
                worker.start()

                # Store the pipe ends we need
                self.worker_assign_fds.append(worker.assign_w)
                self.worker_done_fds.append(worker.done_r)
                
                # Register the done pipe with the selector
                self.sel.register(worker.done_r, selectors.EVENT_READ, self._process_worker_done)
                os.set_blocking(worker.done_r, False)
                
                self.workers.append(worker)
                self.available_workers.append(wid)

            # Wait for the workers to start up
            time.sleep(0.1)

            # Spawn the writer process
            writer = Writer(
                shm_name=self.shm.name,
                sample_rate=self.sample_rate,
                chunk_size=self.chunk_size,
                outbuf_num_chunks=self.pool_size,
                pool_size=self.pool_size,
                device_name=self.device_name,
                ao_channels=self.ao_seqs,
                do_channels=[]  # Placeholder for digital output channels
            )
            writer.daemon = True
            writer.start()
            
            # Store the pipe ends we need
            self.chunk_ready_fd = writer.ready_w
            self.slot_free_fd = writer.report_r
            
            # Register the writer's pipe with the selector
            self.sel.register(self.slot_free_fd, selectors.EVENT_READ, self._process_slot_free)
            os.set_blocking(self.slot_free_fd, False)
            
            self.writer = writer

            # Start the manager loop
            print(f"Starting streaming of {self.num_chunks_to_stream} chunks on {self.num_channels} channels...")
            self._manager_loop()

        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt, cleaning up...")
        except Exception as e:
            print(f"Error in sequence streamer: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.cleanup()

    def __del__(self):
        """Destructor - safety net for cleanup."""
        self.cleanup()

    def cleanup(self):
        """Clean up all resources."""
        # Stop all processes
        for worker in self.workers:
            worker.terminate()
            worker.join(timeout=1.0)

        self.writer.terminate()
        self.writer.join(timeout=1.0)

        # Close shared memory
        try:
            self.shm.close()
            self.shm.unlink()  # This will delete the shared memory segment
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error unlinking shared memory: {e}")

        # Close selector
        self.sel.close()


        
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

