# SequenceStreamer.py
# Created by: Marcin Kalinowski & Yi Zhu

import os
import time
import struct
import numpy as np
import selectors

from Channels import AnalogChannel, DigitalChannel

from Worker import Worker
from Writer import Writer, TEST_MODE

import multiprocessing
multiprocessing.set_start_method("fork")


class SequenceStreamer:
    """
    Manager class: sets up shared memory, manages worker and writer processes,
    and runs the even loop in the main thread to assign chunks and collect reports.
    """

    WORKER_ASSIGN_STRUCT = struct.Struct('>q i I I')  # (seq, buf_idx, ch_start, ch_end)
    WORKER_DONE_STRUCT = struct.Struct('>q i I d')  # (seq, buf_idx, worker_id, compute_time)
    CHUNK_READY_STRUCT = struct.Struct('>q i')  # (seq, buf_idx)
    SLOT_FREE_STRUCT = struct.Struct('>i')  # (buf_idx)

    def __init__(
            self,
            num_channels: int,  # Total number of channels
            chunk_size: int,  # Size of each chunk in samples
            pool_size: int,  # Size of the memory pool
            sample_rate: float,  # Sample rate in Hz
            num_workers: int,  # Number of worker processes
            device_name: str,  # Name of the DAQ device
    ) -> None:
        self.num_channels = num_channels
        self.chunk_size = chunk_size
        self.pool_size = pool_size
        self.sample_rate = sample_rate
        self.num_workers = num_workers
        self.device_name = device_name

        # Create shared memory segment for the data buffer
        shm_size = pool_size * num_channels * chunk_size * np.dtype(np.float64).itemsize
        shm_name = f"nistreamer_shared_memory_{os.getpid()}"

        # Create shared memory segment
        self.shm = multiprocessing.shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)
        self.buffer = np.ndarray(
            (pool_size, num_channels, chunk_size), 
            dtype=np.float64, 
            buffer=self.shm.buf
        )
        self.buffer.fill(0.0)

        print(f"Shared memory segment '{shm_name}' created with size {shm_size} bytes: ({pool_size}x [{chunk_size} chunk size for {num_channels} channels]).")

        # Initialize file descriptors for inter-process communication with the writer.
        self.chunk_ready_child_fd, self.chunk_ready_fd = os.pipe()
        self.slot_free_fd, self.slot_free_child_fd = os.pipe()

        # Initialize file descriptors for inter-process communication with the workers.
        self.worker_assign_fds = []
        self.worker_done_fds = []

        self.worker_assign_child_fds = []
        self.worker_done_child_fds = []

        # Compute channel ranges
        channels_per_worker = num_channels // num_workers
        self.channel_ranges = []
        for i in range(num_workers-1):
            self.channel_ranges.append((i*channels_per_worker, (i+1)*channels_per_worker))
        # The last worker gets the remaining channels
        self.channel_ranges.append(((num_workers-1)*channels_per_worker, num_channels))

        # Placeholder for processes
        self.workers = []
        self.writer = None

        # Manager internal state
        self.completed_chunks = []
        self.next_calc_seq = 0
        self.next_calc_ch_group = 0
        self.next_streamed_seq = 0
        self.ema_alpha = 0.3
        self.available_workers = []
        self.available_slots = [i for i in range(self.pool_size)]  # List of available buffer slots
        self.processed_seqs = {} # Dictionary of of the form {seq: set(remaining channel groups)]}
        self.processed_seqs_slot = {} # Keep track of which sequence is processed in which slots (sanity check)

        # Selector for worker and free pipes
        self.sel = selectors.DefaultSelector()

        # Set non-blocking mode for manager-readable pipes
        os.set_blocking(self.slot_free_fd, False)

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
        if hasattr(self, 'workers'):
            for worker in self.workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=1.0)
                    if worker.is_alive():
                        print(f"Warning: Worker {worker.worker_id} did not terminate gracefully")
                        worker.kill()

        if hasattr(self, 'writer') and self.writer is not None:
            if self.writer.is_alive():
                self.writer.terminate()
                self.writer.join(timeout=1.0)
                if self.writer.is_alive():
                    print("Warning: Writer did not terminate gracefully")
                    self.writer.kill()

        # Close all pipes
        for fd in getattr(self, 'worker_assign_fds', []):
            try:
                os.close(fd)
            except OSError:
                pass

        for fd in getattr(self, 'worker_done_fds', []):
            try:
                os.close(fd)
            except OSError:
                pass

        for fd in getattr(self, 'worker_assign_child_fds', []):
            try:
                os.close(fd)
            except OSError:
                pass

        for fd in getattr(self, 'worker_done_child_fds', []):
            try:
                os.close(fd)
            except OSError:
                pass

        # Close writer pipes
        for attr in ['chunk_ready_fd', 'slot_free_fd', 'chunk_ready_child_fd', 'slot_free_child_fd']:
            if hasattr(self, attr):
                try:
                    os.close(getattr(self, attr))
                except OSError:
                    pass

        # Close shared memory
        if hasattr(self, 'shm'):
            try:
                self.shm.close()
                self.shm.unlink()  # This will delete the shared memory segment
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Error cleaning up shared memory: {e}")

        # Close selector
        if hasattr(self, 'sel'):
            self.sel.close()

    def _process_worker_done(self, fd):
        """
        Process a worker done notification.
        This method reads the sequence number and buffer index from the pipe,
        marks the slot as available, and updates the processed sequences.
        """
        data = os.read(fd, self.WORKER_DONE_STRUCT.size)
        if len(data) < self.WORKER_DONE_STRUCT.size:
            raise ValueError("Received incomplete worker done notification.")
        seq, buf_idx, worker_id, compute_time = self.WORKER_DONE_STRUCT.unpack(data)
        print(f"Received done notification: seq={seq}, buf_idx={buf_idx}, worker_id={worker_id}, compute_time={compute_time}")

        if seq not in self.processed_seqs:
            raise ValueError(f"Received done notification for unknown sequence {seq}.")
        
        if seq not in self.processed_seqs_slot:
            raise ValueError(f"Received done notification for unknown sequence {seq} in slots.")
        
        if self.processed_seqs_slot[seq] != buf_idx:
            raise ValueError(f"Received done notification for sequence {seq} in unexpected slot {buf_idx} vs {self.processed_seqs_slot[seq]}.")

        # Free the worker
        self.available_workers.append(worker_id)

        # Decrement the number of processed channel groups for this sequence
        self.processed_seqs[seq] -= 1

        # All channel groups for this sequence have been processed
        if self.processed_seqs[seq] == 0:
            # Remove the sequence from processed sequences
            del self.processed_seqs[seq]
            del self.processed_seqs_slot[seq]

            # Write the chunk ready notification to the writer
            self._assign_slot_write(seq, buf_idx)

    def _process_slot_free(self, fd):
        """
        Process a slot free notification from the writer.
        This method reads the slot index from the pipe and adds it as available.
        """
        data = os.read(fd, self.SLOT_FREE_STRUCT.size)
        if len(data) < self.SLOT_FREE_STRUCT.size:
            raise ValueError("Received incomplete slot free notification.")
        
        buf_idx = self.SLOT_FREE_STRUCT.unpack(data)[0]

        if buf_idx == -1:
            raise ValueError("Received error from writer: -1.")

        self.available_slots.append(buf_idx)

        print(f"Slot {buf_idx} freed by writer.")

    def _assign_chunk(self, worker_id: int, seq: int, buf_idx: int, ch_start: int, ch_end: int):
        """Assign a chunk to a worker."""
        print(f"Assigning chunk to worker {worker_id} at assign_w={self.worker_assign_fds[worker_id]}: seq={seq}, buf_idx={buf_idx}, ch_start={ch_start}, ch_end={ch_end}")
        assign_data = self.WORKER_ASSIGN_STRUCT.pack(seq, buf_idx, ch_start, ch_end)
        bytes_written = os.write(self.worker_assign_fds[worker_id], assign_data)

    def _assign_slot_write(self, seq: int, buf_idx: int):
        """
        Notify the writer that a chunk is ready to be written.
        This method writes the sequence number and buffer index to the pipe.
        """
        ready_data = self.CHUNK_READY_STRUCT.pack(seq, buf_idx)
        os.write(self.chunk_ready_fd, ready_data)

    def _manager_loop(self):
        while True:
            # While there are free workers and available slots, assign chunks to be computed
            while len(self.available_workers):
                if self.next_calc_seq in self.processed_seqs:
                    buf_idx = self.processed_seqs_slot[self.next_calc_seq]
                else:
                    if len(self.available_slots):
                        buf_idx = self.available_slots.pop()
                        self.processed_seqs_slot[self.next_calc_seq] = buf_idx
                        self.processed_seqs[self.next_calc_seq] = self.num_workers
                    else:
                        break

                # Get the next available worker
                worker_id = self.available_workers.pop()

                # Determine the channel range for this worker
                ch_start, ch_end = self.channel_ranges[self.next_calc_ch_group]

                # Assign the chunk to the worker
                self._assign_chunk(worker_id, self.next_calc_seq, buf_idx, ch_start, ch_end)

                # Increment the sequence number if necessary
                self.next_calc_ch_group += 1
                if self.next_calc_ch_group >= self.num_workers:
                    self.next_calc_ch_group = 0
                    self.next_calc_seq += 1

            # Check for completed chunks from workers and free slots from writer
            events = self.sel.select(timeout=1e-6)
            for key, _ in events:
                callback = key.data
                callback(key.fileobj)

    def start(self):
        try:
            # Spawn worker processes
            for wid in range(self.num_workers):
                worker = Worker(
                    worker_id=wid,
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

            # Create analog channels
            ao_channels = [AnalogChannel(name=f"{self.device_name}/ao{i}", min_val=-2.0, max_val=2.0) for i in range(self.num_channels)]

            # Spawn the writer process
            writer = Writer(
                shm_name=self.shm.name,
                sample_rate=self.sample_rate,
                chunk_size=self.chunk_size,
                outbuf_num_chunks=self.pool_size,
                pool_size=self.pool_size,
                num_channels=self.num_channels,
                device_name=self.device_name,
                ao_channels=ao_channels,
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
            self._manager_loop()
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt, cleaning up...")
        except Exception as e:
            print(f"Error in sequence streamer: {e}")
        finally:
            self.cleanup()


        
if __name__ == "__main__":
    # Example usage with context manager
    with SequenceStreamer(
        num_channels=8,  # Total number of channels
        chunk_size=65536,  # Size of each chunk in samples
        pool_size=4,  # Size of the memory pool
        sample_rate=1e6,  # Sample rate in Hz
        num_workers=4,  # Number of worker processes
        device_name="PXI1Slot3"  # Name of the DAQ device
    ) as streamer:
        streamer.start()

