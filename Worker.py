# Worker.py
# Created by: Marcin Kalinowski & Yi Zhu

import os
import struct
import numpy as np
import time
from multiprocessing import Process, shared_memory

class Worker(Process):
    """
    Worker process: reads chunk assignments via a pipe, compute its subset of channels
    into shared memory, and reports results back via another pipe.
    """

    ASSIGN_STRUCT = struct.Struct('>q i I I')  # (seq, buf_idx, ch_start, ch_end)
    REPORT_STRUCT = struct.Struct('>q i I d')  # (seq, buf_idx, worker_id, compute_time)

    def __init__(
            self,
            worker_id: int,  # Worker ID for identification
            shm_name: str,  # Name of the shared memory segment
            sample_rate: float,  # Sample rate in Hz
            chunk_size: int,  # Size of each chunk in samples
            num_channels: int,  # Total number of channels
            pool_size: int,  # Size of the memory pool
            ) -> None:
        super().__init__(name=f"Worker-{worker_id}")
        self.worker_id = worker_id
        self.shm_name = shm_name
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.num_channels = num_channels
        self.pool_size = pool_size

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
                print(f"Worker {self.worker_id} waiting for assignment...")
                # Block until we get exactly 16 bytes of assignment data
                assign_data = os.read(self.assign_r, self.ASSIGN_STRUCT.size)
                if not assign_data:
                    print(f"Worker {self.worker_id} got EOF")
                    break

                # Parse data
                if len(assign_data) < self.ASSIGN_STRUCT.size:
                    raise ValueError(f"Received incomplete assignment data in Worker({self.worker_id}).")
                
                seq, buf_idx, ch_start, ch_end = self.ASSIGN_STRUCT.unpack(assign_data)
                print(f"Worker {self.worker_id} received assignment: seq={seq}, buf_idx={buf_idx}, ch_start={ch_start}, ch_end={ch_end}")

                # Start the timer for computation
                compute_time = time.time()

                # Compute absolute sample indices for this chunk
                base_idx = seq * self.chunk_size

                # Compute time vector in seconds
                t = np.arange(base_idx, base_idx + self.chunk_size) / self.sample_rate

                T0, T1 = seq * self.chunk_size, (seq + 1) * self.chunk_size

                # Time generation of the assigned channels
                for ch in range(ch_start, ch_end):
                    inst_set = self.inst_set[ch]
                    in_chunk_idx = 0
                    while self.current_inst_idx[ch] < len(inst_set):
                        # Get the current instruction
                        (ins_t0,ins_t1) = inst_set[self.current_inst_idx[ch]].times
                        func = inst_set[self.current_inst_idx[ch]].func

                        # Calculate the time within the instruction
                        t = np.arange(max(ins_t0, T0)-T0, min(ins_t1, T1)-T0) / self.sample_rate

                        # Calculate the indices of the time within the buffer
                        buf_st, buf_end = in_chunk_idx, in_chunk_idx + len(t)

                        # Write the data to the buffer
                        self.buffer[buf_idx, ch, buf_st:buf_end] = func(t)

                        # Move to the next instruction
                        self.current_inst_idx[ch] += 1

                        # Increment the in-chunk index
                        in_chunk_idx += len(t)

                        # If the current instruction ends after the chunk boundary, break
                        if ins_t1 > T1:
                            break

                # End the timer for computation
                compute_time = time.time() - compute_time

                # Prepare report data
                report_msg = self.REPORT_STRUCT.pack(seq, buf_idx, self.worker_id, compute_time)

                # Send report back to the main process
                os.write(self.done_w, report_msg)

        except Exception as e:
            print(f"Worker {self.worker_id} failed: {e}")
        finally:
            # Cleanup
            if self.shm:
                self.shm.close()