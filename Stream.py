import numpy as np
from typing import Dict, List, Tuple, NamedTuple
from multiprocessing import Process, Value, shared_memory, Lock
import ctypes
from AOCard import AOCard
import os
import time

class ChannelData(NamedTuple):
    """Data structure for channel information."""
    default_val: float
    instructions: list
    is_compiled: bool
    chunk_instructions: List[List[int]]

def create_shared_buffer(shape, dtype=np.float64):
    """Create a shared memory buffer."""
    size = int(np.prod(shape))
    nbytes = size * np.dtype(dtype).itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    return np.ndarray(shape, dtype=dtype, buffer=shm.buf), shm

def worker_process(channel_data_list: List[Tuple[ChannelData, int]], 
                  chunk_size: int,
                  current_chunk: Value,
                  running: Value,
                  process_id: int,
                  shared_buffer_name: str,
                  sync_buffer_name: str,
                  sync_lock: Lock):
    """Worker process that continuously processes chunks."""
    
    # Set process affinity
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {process_id % os.cpu_count()})
    
    # Connect to shared memory buffers
    data_shm = shared_memory.SharedMemory(name=shared_buffer_name)
    sync_shm = shared_memory.SharedMemory(name=sync_buffer_name)
    
    # Create array views
    n_channels = len(channel_data_list)
    output_buffer = np.ndarray((n_channels, chunk_size), dtype=np.float64, buffer=data_shm.buf)
    sync_array = np.ndarray((2,), dtype=np.int32, buffer=sync_shm.buf)  # [completion_count, barrier]
    
    # Pre-calculate time offsets and views
    time_offsets = np.arange(chunk_size, dtype=np.float64)
    channel_views = [output_buffer[i] for i in range(n_channels)]
    
    # Pre-allocate temporary arrays
    temp_array = np.empty(chunk_size, dtype=np.float64)
    
    # Track last processed chunk
    last_processed_chunk = -1
    
    while running.value:
        # Get current chunk number
        chunk_idx = current_chunk.value
        
        # Skip if we've already processed this chunk
        if chunk_idx == last_processed_chunk:
            continue
            
        # Calculate chunk start sample
        chunk_start = chunk_idx * chunk_size
        
        # Process each channel
        for i, (channel_data, _) in enumerate(channel_data_list):
            # Get pre-calculated output view
            channel_data_view = channel_views[i]
            
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
                    temp_view = temp_array[rel_start:rel_end]
                    
                    # Apply instruction
                    if type_ == 0:  # CONST
                        output_view.fill(params['val'])
                    elif type_ == 1:  # LINRAMP
                        np.multiply(T, params['A'], out=temp_view)
                        np.add(temp_view, params['B'], out=output_view)
                    elif type_ == 2:  # SINE
                        np.multiply(T + chunk_start + rel_start, params['omega'], out=temp_view)
                        np.add(temp_view, params['phase'], out=temp_view)
                        np.sin(temp_view, out=temp_view)
                        np.multiply(temp_view, params['A'], out=temp_view)
                        np.add(temp_view, params['offset'], out=output_view)
        
        # Update last processed chunk and notify completion
        last_processed_chunk = chunk_idx
        
        # Atomic increment of completion counter
        with sync_lock:
            sync_array[0] += 1
        
        # Wait at barrier with exponential backoff
        barrier = sync_array[1]
        backoff = 0.000001  # 1 microsecond initial backoff
        while running.value and current_chunk.value == chunk_idx and sync_array[1] == barrier:
            time.sleep(backoff)
            backoff = min(backoff * 2, 0.001)  # Double backoff up to 1ms max
    
    # Cleanup
    data_shm.close()
    sync_shm.close()

class Stream:
    """Class for streaming analog output data in chunks."""
    
    def __init__(self, card: AOCard, chunk_size: int = 1000, channels_per_process: int = 8):
        """Initialize the stream with an AOCard and chunk size."""
        self.card = card
        self.chunk_size = int(chunk_size)
        self.current_chunk = Value(ctypes.c_longlong, 0)
        self.running = Value(ctypes.c_bool, True)
        self.sync_lock = Lock()
        
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
        
        # Create shared memory buffers
        n_channels = len(card.channels)
        self.output_buffer, self.data_shm = create_shared_buffer((n_channels, self.chunk_size))
        
        # Create synchronization buffer [completion_count, barrier]
        self.sync_buffer, self.sync_shm = create_shared_buffer((2,), dtype=np.int32)
        self.sync_buffer.fill(0)
        
        # Pre-calculate channel batches
        self.channel_batches = []
        for i in range(0, n_channels, channels_per_process):
            batch_channels = list(card.channels.keys())[i:i + channels_per_process]
            batch_data = [(self.channel_data[ch_num], ch_num) for ch_num in batch_channels]
            self.channel_batches.append(batch_data)
        
        # Start worker processes
        self.processes = []
        for i, batch_data in enumerate(self.channel_batches):
            p = Process(
                target=worker_process,
                args=(batch_data, self.chunk_size, self.current_chunk, 
                      self.running, i, self.data_shm.name, self.sync_shm.name,
                      self.sync_lock)
            )
            p.start()
            self.processes.append(p)
    
    def __del__(self):
        """Clean up processes and shared memory."""
        try:
            # Stop all processes
            if hasattr(self, 'running'):
                self.running.value = False
            
            # Terminate and clean up processes
            if hasattr(self, 'processes'):
                for p in self.processes:
                    try:
                        if p.is_alive():
                            p.terminate()
                        p.join(timeout=0.1)
                        if p.is_alive():
                            p.kill()
                    except:
                        pass
            
            # Clean up shared memory
            if hasattr(self, 'data_shm'):
                try:
                    self.data_shm.close()
                    self.data_shm.unlink()
                except:
                    pass
                    
            if hasattr(self, 'sync_shm'):
                try:
                    self.sync_shm.close()
                    self.sync_shm.unlink()
                except:
                    pass
                    
        except:
            pass
    
    def calc_next_chunk(self) -> np.ndarray:
        """Calculate the next chunk of samples for all channels."""
        if not self.channel_data:
            return np.array([])
        
        # Reset completion counter with atomic operation
        with self.sync_lock:
            self.sync_buffer[0] = 0
        
        # Wait for all processes to complete their work with exponential backoff
        n_processes = len(self.channel_batches)
        backoff = 0.000001  # 1 microsecond initial backoff
        while self.sync_buffer[0] < n_processes:
            time.sleep(backoff)
            backoff = min(backoff * 2, 0.001)  # Double backoff up to 1ms max
        
        # Signal processes to continue by incrementing barrier
        self.sync_buffer[1] += 1
        
        # Update chunk index for next chunk
        self.current_chunk.value += 1
        return self.output_buffer 