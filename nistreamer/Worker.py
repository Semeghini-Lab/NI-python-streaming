# Worker.py
# Created by: Marcin Kalinowski & Yi Zhu

import struct
import numpy as np
import time
import bisect
from multiprocessing import Process, shared_memory
from nistreamer.Sequences import AOSequence, DOSequence
from nistreamer.NICard import NICard

import zmq

class Worker(Process):
    """
    Worker process: receives chunk assignments via ZMQ, computes the data,
    and reports results back via ZMQ.
    """

    ASSIGN_STRUCT = struct.Struct('>q i I I I')  # (chunk_idx, buf_idx, card_idx, ch_start, ch_end)
    DONE_STRUCT = struct.Struct('>q i I I d')  # (chunk_idx, buf_idx, card_idx, worker_id, compute_time)

    def __init__(
            self,
            worker_id: int,  # ID of this worker
            cards: list[NICard], # List of cards to work on
            pool_size: int,  # Size of the memory pool for calculations
            assign_port: int,  # Port for receiving assignments
            done_port: int,  # Port for sending completion reports
    ) -> None:
        super().__init__(name=f"Worker-{worker_id}")
        self.worker_id = worker_id
        self.cards = cards
        self.pool_size = pool_size

        # Initialize the instruction sets for each channel for each card
        self.ins_set = [[ch.instructions for ch in card.sequences] for card in self.cards]
        self.ins_starts = [[[ins.start_sample for ins in ch.instructions] for ch in card.sequences] for card in self.cards]
        self.ins_inplaces = [[[ins.inplace for ins in ch.instructions] for ch in card.sequences] for card in self.cards]

        # Extract the chunk size for each card
        self.chunk_sizes = [card.chunk_size for card in self.cards]

        # Extract the sample rate for each card
        self.sample_rates = [card.sample_rate for card in self.cards]

        # Save ports for communication
        self.assign_port = assign_port
        self.done_port = done_port

        # These will be set in run()
        self.buffers = None
        self.assign_socket = None
        self.done_socket = None
        self.shms = None  # Store shared memory objects

    def run(self):
        try:
            # Create ZMQ context and sockets
            context = zmq.Context()
            # Worker receives assignments
            self.assign_socket = context.socket(zmq.PULL)
            # Worker sends completion reports
            self.done_socket = context.socket(zmq.PUSH)
            
            # Connect to ports
            self.assign_socket.connect(f"tcp://127.0.0.1:{self.assign_port}")
            self.done_socket.connect(f"tcp://127.0.0.1:{self.done_port}")

            # Create shared memory objects and buffers
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

            # Main worker loop
            while True:
                # Wait for assignment
                data = self.assign_socket.recv()
                chunk_idx, buf_idx, card_idx, ch_start, ch_end = self.ASSIGN_STRUCT.unpack(data)

                if chunk_idx == -1:
                    print(f"Worker {self.worker_id}: received stop message.")
                    break

                # Start the timer for computation
                compute_time = time.time()

                # Compute the ends of the chunk in sample indices
                chunk_start, chunk_end = chunk_idx * self.chunk_sizes[card_idx], (chunk_idx + 1) * self.chunk_sizes[card_idx]

                # Time generation of the assigned channels
                for ch in range(ch_start, ch_end):
                    # Get the instruction set for the card and channel
                    ins_set = self.ins_set[card_idx][ch]
                    ins_starts = self.ins_starts[card_idx][ch]
                    ins_inplaces = self.ins_inplaces[card_idx][ch]

                    # Find the first instruction start time that crosses the chunk start boundary
                    ins_idx = bisect.bisect_right(ins_starts, chunk_start)-1
                    if ins_idx < 0:
                        raise ValueError(f"Worker {self.worker_id} encountered an error while processing channel {ch} at chunk start {chunk_start}.")

                    # Start the counter within the chunk
                    in_chunk_pos = 0
                    while in_chunk_pos < self.chunk_sizes[card_idx]: # This relies on chunk alignment at compile time
                        # Get the current instruction
                        (ins_start,ins_end) = ins_set[ins_idx].start_sample, ins_set[ins_idx].end_sample
                        
                        # Get the function of the current instruction
                        func = ins_set[ins_idx].func

                        # Calculate the time within the instruction
                        t = np.arange(max(ins_start, chunk_start)-ins_start, min(ins_end, chunk_end)-ins_start) / self.sample_rates[card_idx]

                        # Write the data to the buffer
                        if ins_inplaces[ins_idx]:
                            #print(self.buffers[card_idx].itemsize)
                            func(t, buf=np.ndarray(
                                shape = (len(t),),
                                dtype = self.buffers[card_idx].dtype,
                                buffer = self.buffers[card_idx][buf_idx, ch, in_chunk_pos:in_chunk_pos+len(t)].data
                            ))
                        else:
                            self.buffers[card_idx][buf_idx, ch, in_chunk_pos:in_chunk_pos+len(t)] = func(t)

                        # Move to the next instruction
                        ins_idx += 1

                        # Increment the in-chunk index
                        in_chunk_pos += len(t)

                # End the timer for computation
                compute_time = time.time() - compute_time

                # Prepare done message
                done_msg = self.DONE_STRUCT.pack(chunk_idx, buf_idx, card_idx, self.worker_id, compute_time)

                # Send done message back to the main process
                self.done_socket.send(done_msg)

        except Exception as e:
            print(f"[ERROR] Worker {self.worker_id}: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Clean up all the resources specific to the Worker.
        Shared memory will be closed by the manager.
        """
        pass