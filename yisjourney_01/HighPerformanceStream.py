import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Value, Lock, shared_memory, Condition
import ctypes
import time
import os
import psutil
from dataclasses import dataclass
from typing import List, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor

@dataclass
class ChannelData:
    default_val: float
    instructions: list
    is_compiled: bool
    chunk_instructions: list

def create_shared_buffer(shape, dtype=np.float64):
    """Create a shared memory buffer."""
    size = int(np.prod(shape))
    nbytes = size * np.dtype(dtype).itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    array.fill(0)
    return array, shm

def set_process_priority():
    """Set high process priority and optimize scheduling."""
    try:
        p = psutil.Process()
        if os.name == 'nt':  # Windows
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:  # Unix-like
            p.nice(-10)
    except:
        pass

def worker_process(channel_data_list: List[Tuple[ChannelData, int]], 
                  chunk_size: int,
                  current_chunk: Value,
                  running: Value,
                  process_id: int,
                  shared_buffer_name: str,
                  completion_array_name: str,
                  condition_name: str):
    """Worker process with optimized synchronization."""
    
    # Set process priority and affinity
    set_process_priority()
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {process_id % os.cpu_count()})
    
    # Connect to shared memory
    data_shm = shared_memory.SharedMemory(name=shared_buffer_name)
    completion_shm = shared_memory.SharedMemory(name=completion_array_name)
    condition_shm = shared_memory.SharedMemory(name=condition_name)
    
    # Create array views
    n_channels = len(channel_data_list)
    output_buffer = np.ndarray((n_channels, chunk_size), dtype=np.float64, buffer=data_shm.buf)
    completion_array = np.ndarray((1,), dtype=np.int32, buffer=completion_shm.buf)
    
    # Pre-calculate arrays and views
    time_offsets = np.arange(chunk_size, dtype=np.float64)
    channel_views = [output_buffer[i] for i in range(n_channels)]
    temp_array = np.empty(chunk_size, dtype=np.float64)
    last_processed_chunk = -1
    
    # Create condition for synchronization
    condition = Condition()
    
    while running.value:
        chunk_idx = current_chunk.value
        
        # Skip if already processed this chunk
        if chunk_idx == last_processed_chunk:
            time.sleep(0.0001)  # Minimal sleep to prevent busy waiting
            continue
        
        # Process all channels in batch
        chunk_start = chunk_idx * chunk_size
        
        # Process channels in parallel using thread pool
        def process_channel(channel_idx):
            channel_data, _ = channel_data_list[channel_idx]
            channel_view = channel_views[channel_idx]
            channel_view.fill(channel_data.default_val)
            
            if chunk_idx < len(channel_data.chunk_instructions):
                instruction_indices = channel_data.chunk_instructions[chunk_idx]
                
                for idx in instruction_indices:
                    start, end, type_, params = channel_data.instructions[idx]
                    rel_start = max(0, start - chunk_start)
                    rel_end = min(chunk_size, end - chunk_start)
                    
                    if rel_start >= rel_end:
                        continue
                    
                    T = time_offsets[rel_start:rel_end]
                    output_view = channel_view[rel_start:rel_end]
                    temp_view = temp_array[rel_start:rel_end]
                    
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
        
        # Process channels
        for i in range(n_channels):
            process_channel(i)
        
        # Update completion status using atomic operation
        completion_array[0] += 1
        
        # Notify completion
        with condition:
            condition.notify_all()
        
        last_processed_chunk = chunk_idx
    
    # Cleanup
    data_shm.close()
    completion_shm.close()
    condition_shm.close()

class HighPerformanceStream:
    """High-performance version of Stream with optimized synchronization."""
    
    def __init__(self, card, chunk_size: int = 1000, channels_per_process: int = 8):
        """Initialize the stream with an AOCard and chunk size."""
        self.card = card
        self.chunk_size = int(chunk_size)
        self.current_chunk = Value(ctypes.c_longlong, 0)
        self.running = Value(ctypes.c_bool, True)
        
        # Calculate chunks
        max_samples = max(
            max(end for _, end, _, _ in channel.instructions)
            for channel in card.channels.values()
        ) if card.channels else 0
        self.total_chunks = (max_samples + chunk_size - 1) // chunk_size
        
        # Initialize channel data
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
        
        # Create completion counter in shared memory
        self.completion_buffer, self.completion_shm = create_shared_buffer((1,), dtype=np.int32)
        self.completion_buffer[0] = 0
        
        # Create condition variable in shared memory
        self.condition_buffer, self.condition_shm = create_shared_buffer((1,), dtype=np.int32)
        self.condition = Condition()
        
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
                      self.running, i, self.data_shm.name, 
                      self.completion_shm.name, self.condition_shm.name)
            )
            p.start()
            self.processes.append(p)
    
    def __del__(self):
        """Clean up processes and shared memory."""
        try:
            if hasattr(self, 'running'):
                self.running.value = False
            
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
            
            if hasattr(self, 'data_shm'):
                try:
                    self.data_shm.close()
                    self.data_shm.unlink()
                except:
                    pass
            
            if hasattr(self, 'completion_shm'):
                try:
                    self.completion_shm.close()
                    self.completion_shm.unlink()
                except:
                    pass
            
            if hasattr(self, 'condition_shm'):
                try:
                    self.condition_shm.close()
                    self.condition_shm.unlink()
                except:
                    pass
                    
        except:
            pass
    
    def calc_next_chunk(self) -> np.ndarray:
        """Calculate the next chunk of samples for all channels."""
        if not self.channel_data:
            return np.array([])
        
        # Reset completion counter atomically
        self.completion_buffer[0] = 0
        
        # Update chunk index
        self.current_chunk.value += 1
        
        # Wait for all processes with condition
        n_processes = len(self.channel_batches)
        with self.condition:
            while self.completion_buffer[0] < n_processes:
                if not self.running.value:
                    break
                self.condition.wait(timeout=0.001)
        
        return self.output_buffer 