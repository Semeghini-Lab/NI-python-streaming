# Worker.py
# Created by: Marcin Kalinowski & Yi Zhu

import os
import struct
import numpy as np
import time
import bisect
from multiprocessing import Process, shared_memory
from AOSequence import AOSequence

class Worker(Process):
    """
    Worker process: reads chunk assignments via a pipe, compute its subset of channels
    into shared memory, and reports results back via another pipe.
    """

    ASSIGN_STRUCT = struct.Struct('>q i I I')  # (chunk_idx, buf_idx, ch_start, ch_end)
    REPORT_STRUCT = struct.Struct('>q i I d')  # (chunk_idx, buf_idx, worker_id, compute_time)

    def __init__(
            self,
            worker_id: int,  # Worker ID for identification
            shm_name: str,  # Name of the shared memory segment
            sample_rate: float,  # Sample rate in Hz
            chunk_size: int,  # Size of each chunk in samples
            num_channels: int,  # Total number of channels
            ao_channels: list[AOSequence],  # List of analog output channels
            pool_size: int,  # Size of the memory pool
            ) -> None:
        super().__init__(name=f"Worker-{worker_id}")
        self.worker_id = worker_id
        self.shm_name = shm_name
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.num_channels = num_channels
        self.pool_size = pool_size
        self.ao_channels = ao_channels

        # Initialize the instruction sets for each channel
        self.ins_set = [ch.instructions for ch in self.ao_channels]
        self.ins_starts = [[ins.start_sample for ins in ch.instructions] for ch in self.ao_channels]

        # Create all communication pipes in parent
        self.assign_r, self.assign_w = os.pipe()
        self.done_r, self.done_w = os.pipe()

        # These will be set in run()
        self.shm = None
        self.buffer = None

    def run(self):
        try:
            # Attach to shared memory segment
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.buffer = np.ndarray(
                (self.pool_size, self.num_channels, self.chunk_size),
                dtype=np.float64,
                buffer=self.shm.buf
            )

            while True:
                # Block until we get exactly 16 bytes of assignment data
                assign_data = os.read(self.assign_r, self.ASSIGN_STRUCT.size)
                if not assign_data:
                    raise EOFError(f"Worker {self.worker_id} got EOF")

                # Parse data
                if len(assign_data) < self.ASSIGN_STRUCT.size:
                    raise ValueError(f"Received incomplete assignment data in Worker({self.worker_id}).")
                
                chunk_idx, buf_idx, ch_start, ch_end = self.ASSIGN_STRUCT.unpack(assign_data)
                print(f"Worker {self.worker_id} received assignment: chunk_idx={chunk_idx}, buf_idx={buf_idx}, ch_start={ch_start}, ch_end={ch_end}")

                if chunk_idx == -1:
                    print(f"Worker {self.worker_id} received stop message.")
                    break

                # Start the timer for computation
                compute_time = time.time()

                # Compute the ends of the chunk in sample indices
                chunk_start, chunk_end = chunk_idx * self.chunk_size, (chunk_idx + 1) * self.chunk_size

                # Time generation of the assigned channels
                for ch in range(ch_start, ch_end):
                    # Get the instruction set for the channel
                    ins_set = self.ins_set[ch]
                    ins_starts = self.ins_starts[ch]

                    # Find the first instruction start time that crosses the chunk start boundary
                    ins_idx = bisect.bisect_right(ins_starts, chunk_start)-1
                    if ins_idx < 0:
                        raise ValueError(f"Worker {self.worker_id} encountered an error while processing channel {ch} at chunk start {chunk_start}.")

                    # Start the counter within the chunk
                    in_chunk_pos = 0
                    while in_chunk_pos < self.chunk_size: # This relies on chunk alignment at compile time
                        # Get the current instruction
                        (ins_start,ins_end) = ins_set[ins_idx].start_sample, ins_set[ins_idx].end_sample
                        
                        # Get the function of the current instruction
                        func = ins_set[ins_idx].func

                        # Calculate the time within the instruction
                        t = np.arange(max(ins_start, chunk_start)-ins_start, min(ins_end, chunk_end)-ins_start) / self.sample_rate

                        # Write the data to the buffer
                        self.buffer[buf_idx, ch, in_chunk_pos:in_chunk_pos+len(t)] = func(t)

                        # Move to the next instruction
                        ins_idx += 1

                        # Increment the in-chunk index
                        in_chunk_pos += len(t)

                # End the timer for computation
                compute_time = time.time() - compute_time

                # Prepare report data
                report_msg = self.REPORT_STRUCT.pack(chunk_idx, buf_idx, self.worker_id, compute_time)

                # Send report back to the main process
                os.write(self.done_w, report_msg)

        except Exception as e:
            print(f"Worker {self.worker_id}: {e}")
        finally:
            # Cleanup
            self.cleanup()

    def cleanup(self):
        """
        Clean up all the resources specific to the Worker.
        Shared memory will be closed by the manager.
        """
        try:
            # Close pipes
            os.close(self.assign_r)
            os.close(self.assign_w)
            os.close(self.done_r)
            os.close(self.done_w)

        except Exception as e:
            print(f"Error during cleanup: {e}")