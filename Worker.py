# Worker.py
# Created by: Marcin Kalinowski & Yi Zhu

import os
import struct
import numpy as np
import time
import bisect
from multiprocessing import Process, shared_memory
from Sequences import AOSequence, DOSequence
import zmq

class Worker(Process):
    """
    Worker process: receives chunk assignments via ZMQ, computes the data,
    and reports results back via ZMQ.
    """

    ASSIGN_STRUCT = struct.Struct('>q i I I')  # (chunk_idx, buf_idx, ch_start, ch_end)
    DONE_STRUCT = struct.Struct('>q i I d')  # (chunk_idx, buf_idx, worker_id, compute_time)

    def __init__(
            self,
            worker_id: int,  # ID of this worker
            ao_channels: list[AOSequence],  # List of analog output channel named tuples
            shm_name: str,  # Name of the shared memory segment
            sample_rate: float,  # Sample rate in Hz
            chunk_size: int,  # Size of each chunk in samples
            num_channels: int,  # Total number of channels
            pool_size: int,  # Size of the memory pool for calculations
            assign_port: int,  # Port for receiving assignments
            done_port: int,  # Port for sending completion reports
    ) -> None:
        super().__init__(name=f"Worker-{worker_id}")
        self.worker_id = worker_id
        self.ao_channels = ao_channels
        self.shm_name = shm_name
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.num_channels = num_channels
        self.pool_size = pool_size
        self.ao_channels = ao_channels

        # Initialize the instruction sets for each channel
        self.ins_set = [ch.instructions for ch in self.ao_channels]
        self.ins_starts = [[ins.start_sample for ins in ch.instructions] for ch in self.ao_channels]

        # Save ports for communication
        self.assign_port = assign_port
        self.done_port = done_port

        # These will be set in run()
        self.shm = None
        self.buffer = None
        self.context = None
        self.assign_socket = None
        self.done_socket = None

    def run(self):
        try:
            # Create ZMQ context and sockets
            self.context = zmq.Context()
            # Worker receives assignments
            self.assign_socket = self.context.socket(zmq.PULL)
            # Worker sends completion reports
            self.done_socket = self.context.socket(zmq.PUSH)
            
            # Connect to ports
            self.assign_socket.connect(f"tcp://127.0.0.1:{self.assign_port}")
            self.done_socket.connect(f"tcp://127.0.0.1:{self.done_port}")
            
            print(f"Worker {self.worker_id}: Connected to ports assign={self.assign_port}, done={self.done_port}")

            # Attach to shared memory segment
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.buffer = np.ndarray(
                (self.pool_size, self.num_channels, self.chunk_size),
                dtype=np.float64,
                buffer=self.shm.buf
            )

            # Main worker loop
            while True:
                # Wait for assignment
                data = self.assign_socket.recv()
                chunk_idx, buf_idx, ch_start, ch_end = self.ASSIGN_STRUCT.unpack(data)

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

                # Prepare done message
                done_msg = self.DONE_STRUCT.pack(chunk_idx, buf_idx, self.worker_id, compute_time)

                # Send done message back to the main process
                self.done_socket.send(done_msg)

        except Exception as e:
            print(f"Worker {self.worker_id}: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Clean up all the resources specific to the Worker.
        Shared memory will be closed by the manager.
        """
        pass