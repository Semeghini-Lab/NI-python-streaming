# Worker.py
# Created by: Marcin Kalinowski & Yi Zhu

import os
import struct
import numpy as np
import time
from multiprocessing import shared_memory

class Worker:
    """
    Worker process: reads chunk assignments via a pipe, compute its subset of channels
    into shared memory, and reports results back via another pipe.
    """

    ASSIGN_STRUCT = struct.Struct('>I I I I')  # (seq, buf_idx, ch_start, ch_end)
    REPORT_STRUCT = struct.Struct('>I I I d')  # (seq, buf_idx, worker_id, compute_time)

    def __init__(
            self,
            worker_id: int, # Worker ID for identification
            assign_r_fd: int, # Read file descriptor for assignments
            report_w_fd: int, # Write file descriptor for reports
            shm_name: str, # Name of the shared memory segment
            sample_rate: float, # Sample rate in Hz
            chunk_size: int, # Size of each chunk in samples
            num_channels: int, # Total number of channels
            pool_size: int, # Size of the memory pool
            ) -> None:
        self.worker_id = worker_id
        #self.assign_r_fd = assign_r_fd
        #self.report_w_fd = report_w_fd
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.pool_size = pool_size

        # Open the file descriptors for assignments and reports
        self.assign_r_fd = os.fdopen(assign_r_fd, "r")
        self.report_w_fd = os.fdopen(report_w_fd, "w")

        # Attach to shared memory segment
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.buffer = np.ndarray((pool_size, num_channels, chunk_size), dtype=np.float64, buffer=self.shm.buf)

        # Set assignments to blocking mode
        os.set_blocking(assign_r_fd, True)

    def run(self):
        while True:
            # Block until we get exactly 16 bytes of assignment data
            assign_data = os.read(self.assign_r_fd, self.ASSIGN_STRUCT.size)

            # Parse data
            if len(assign_data) < self.ASSIGN_STRUCT.size:
                raise ValueError(f"Received incomplete assignment data in Worker({self.worker_id}).")
            
            seq, buf_idx, ch_start, ch_end = self.ASSIGN_STRUCT.unpack(assign_data)

            # Start the timer for computation
            compute_time = time.time()

            # Compute absolute sample indices for this chunk
            base_idx = seq * self.chunk_size

            time = np.arange(base_idx, base_idx + self.chunk_size) / self.sample_rate

            # Time generation of the assigned channels
            for ch in range(ch_start, ch_end):
                # TODO: Replace with actual data generation logic 
                # TODO: Think about serializing commands.

                # For now, we will generate a simple sine wave for each channel
                self.buffer[buf_idx, ch, :] = np.sin(2 * np.pi * ch * time)

            # End the timer for computation
            compute_time = time.time() - compute_time

            # Prepare report data
            report_msg = self.REPORT_STRUCT.pack(seq, buf_idx, self.worker_id, compute_time)

            # Send report back to the main process
            os.write(self.report_w_fd, report_msg)