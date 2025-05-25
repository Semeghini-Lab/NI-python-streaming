import numpy as np
from typing import Dict, List, Tuple, NamedTuple
from multiprocessing import Process, Queue, Value, shared_memory
import ctypes
from AOCard import AOCard
import os

class ChannelData(NamedTuple):
    """Data structure for channel information."""
    default_val: float
    instructions: list
    is_compiled: bool
    chunk_instructions: List[List[int]]  # List of instruction indices for each chunk

def create_shared_buffer(shape, dtype=np.float64):
    """Create a shared memory buffer."""
    size = int(np.prod(shape))
    nbytes = size * np.dtype(dtype).itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    return np.ndarray(shape, dtype=dtype, buffer=shm.buf), shm

def worker_process(channel_data_list: List[Tuple[ChannelData, int]], 
                  chunk_size: int,
                  result_queue: Queue,
                  current_sample: Value,
                  running: Value,
                  process_id: int,
                  shared_buffer_name: str):
    """Worker process that continuously processes chunks."""
    
    # Set process affinity
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {process_id % os.cpu_count()})
    
    # Connect to shared memory
    shm = shared_memory.SharedMemory(name=shared_buffer_name)
    n_channels = len(channel_data_list)
    output_buffer = np.ndarray((n_channels, chunk_size), dtype=np.float64, buffer=shm.buf)
    
    # Pre-calculate time offsets
    time_offsets = np.arange(chunk_size, dtype=np.float64)
    
    while running.value:
        # Get current sample position and chunk index
        chunk_start = current_sample.value
        chunk_idx = chunk_start // chunk_size
        
        # Process each channel
        for i, (channel_data, channel_num) in enumerate(channel_data_list):
            # Get output view
            channel_data_view = output_buffer[i]
            
            # Fill with default value
            channel_data_view.fill(channel_data.default_val)
            
            # Get pre-calculated instruction indices for this chunk
            if chunk_idx < len(channel_data.chunk_instructions):
                instruction_indices = channel_data.chunk_instructions[chunk_idx]
                
                # Process instructions
                for idx in instruction_indices:
                    start, end, type_, params = channel_data.instructions[idx]
                    
                    # Calculate overlap
                    rel_start = max(0, start - chunk_start)
                    rel_end = min(chunk_size, end - chunk_start)
                    
                    if rel_start >= rel_end:
                        continue
                    
                    # Get views
                    T = time_offsets[rel_start:rel_end]
                    output_view = channel_data_view[rel_start:rel_end]
                    
                    # Apply instruction
                    if type_ == 0:  # CONST
                        output_view.fill(params['val'])
                    elif type_ == 1:  # LINRAMP
                        np.multiply(T, params['A'], out=output_view)
                        output_view += params['B']
                    elif type_ == 2:  # SINE
                        np.multiply(T + chunk_start + rel_start, params['omega'], out=output_view)
                        output_view += params['phase']
                        np.sin(output_view, out=output_view)
                        output_view *= params['A']
                        output_view += params['offset']
        
        # Notify completion
        result_queue.put(process_id)
        
        # Wait for next chunk
        while running.value and current_sample.value == chunk_start:
            pass
    
    # Cleanup
    shm.close()

class Stream:
    """Class for streaming analog output data in chunks."""
    
    def __init__(self, card: AOCard, chunk_size: int = 1000, channels_per_process: int = 8):
        """Initialize the stream with an AOCard and chunk size."""
        self.card = card
        self.chunk_size = int(chunk_size)
        self.current_sample = Value(ctypes.c_longlong, 0)
        self.running = Value(ctypes.c_bool, True)
        
        # Calculate total number of chunks needed
        max_samples = max(
            max(end for _, end, _, _ in channel.instructions)
            for channel in card.channels.values()
        ) if card.channels else 0
        self.total_chunks = (max_samples + chunk_size - 1) // chunk_size
        
        # Initialize channel data with pre-calculated chunk instructions
        self.channel_data = {}
        for channel_num, channel in card.channels.items():
            chunk_instructions = [[] for _ in range(self.total_chunks)]
            for i, instr in enumerate(channel.instructions):
                start_chunk = instr[0] // chunk_size
                end_chunk = (instr[1] + chunk_size - 1) // chunk_size
                for chunk in range(start_chunk, min(end_chunk + 1, self.total_chunks)):
                    chunk_instructions[chunk].append(i)
            
            self.channel_data[channel_num] = ChannelData(
                default_val=channel.default_val,
                instructions=channel.instructions.copy(),
                is_compiled=channel.is_compiled,
                chunk_instructions=chunk_instructions
            )
        
        # Create shared memory buffer
        n_channels = len(card.channels)
        self.output_buffer, self.shared_mem = create_shared_buffer((n_channels, self.chunk_size))
        
        # Pre-calculate channel batches
        self.channel_batches = []
        for i in range(0, n_channels, channels_per_process):
            batch_channels = list(card.channels.keys())[i:i + channels_per_process]
            batch_data = [(self.channel_data[ch_num], ch_num) for ch_num in batch_channels]
            self.channel_batches.append(batch_data)
        
        # Create result queue and start worker processes
        self.result_queue = Queue()
        self.processes = []
        for i, batch_data in enumerate(self.channel_batches):
            p = Process(
                target=worker_process,
                args=(batch_data, self.chunk_size, self.result_queue, 
                      self.current_sample, self.running, i, self.shared_mem.name)
            )
            p.start()
            self.processes.append(p)
    
    def __del__(self):
        """Clean up processes and shared memory."""
        self.running.value = False
        for p in self.processes:
            p.join()
        self.shared_mem.close()
        self.shared_mem.unlink()
    
    def calc_next_chunk(self) -> np.ndarray:
        """Calculate the next chunk of samples for all channels."""
        if not self.channel_data:
            return np.array([])
        
        # Wait for all processes to complete their work
        for _ in range(len(self.channel_batches)):
            self.result_queue.get()
        
        # Update sample position for next chunk
        self.current_sample.value += self.chunk_size
        return self.output_buffer 