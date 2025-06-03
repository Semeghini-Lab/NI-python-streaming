import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Value, Lock, shared_memory
import ctypes
import time
import os
from dataclasses import dataclass
from typing import List, Tuple
import threading

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

def worker_process(channel_data_list: List[Tuple[ChannelData, int]], 
                  chunk_size: int,
                  current_chunk: Value,
                  running: Value,
                  process_id: int,
                  shared_buffer_name: str,
                  sync_buffer_name: str,
                  sync_lock: Lock,
                  event_ready: mp.Event,
                  event_processed: mp.Event):
    """Worker process with improved synchronization."""
    
    # Set process affinity
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {process_id % os.cpu_count()})
    
    # Connect to shared memory buffers
    data_shm = shared_memory.SharedMemory(name=shared_buffer_name)
    sync_shm = shared_memory.SharedMemory(name=sync_buffer_name)
    
    # Create array views
    n_channels = len(channel_data_list)
    output_buffer = np.ndarray((n_channels, chunk_size), dtype=np.float64, buffer=data_shm.buf)
    sync_array = np.ndarray((2,), dtype=np.int32, buffer=sync_shm.buf)
    
    # Pre-calculate time offsets and views
    time_offsets = np.arange(chunk_size, dtype=np.float64)
    channel_views = [output_buffer[i] for i in range(n_channels)]
    temp_array = np.empty(chunk_size, dtype=np.float64)
    last_processed_chunk = -1
    
    while running.value:
        # Wait for ready signal
        event_ready.wait()
        if not running.value:
            break
            
        chunk_idx = current_chunk.value
        if chunk_idx == last_processed_chunk:
            event_ready.clear()
            continue
            
        # Calculate chunk start sample
        chunk_start = chunk_idx * chunk_size
        
        # Process each channel
        for i, (channel_data, _) in enumerate(channel_data_list):
            channel_data_view = channel_views[i]
            channel_data_view.fill(channel_data.default_val)
            
            if chunk_idx < len(channel_data.chunk_instructions):
                instruction_indices = channel_data.chunk_instructions[chunk_idx]
                
                for idx in instruction_indices:
                    start, end, type_, params = channel_data.instructions[idx]
                    rel_start = max(0, start - chunk_start)
                    rel_end = min(chunk_size, end - chunk_start)
                    
                    if rel_start >= rel_end:
                        continue
                    
                    T = time_offsets[rel_start:rel_end]
                    output_view = channel_data_view[rel_start:rel_end]
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
        
        # Update completion status
        last_processed_chunk = chunk_idx
        with sync_lock:
            sync_array[0] += 1
        
        # Signal completion and clear ready flag
        event_processed.set()
        event_ready.clear()
    
    # Cleanup
    data_shm.close()
    sync_shm.close()

class OptimizedStream:
    """Optimized version of Stream class with improved synchronization."""
    
    def __init__(self, card, chunk_size: int = 1000, channels_per_process: int = 8):
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
        self.sync_buffer, self.sync_shm = create_shared_buffer((2,), dtype=np.int32)
        self.sync_buffer.fill(0)
        
        # Create synchronization events
        self.event_ready = mp.Event()
        self.event_processed = mp.Event()
        
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
                      self.sync_lock, self.event_ready, self.event_processed)
            )
            p.start()
            self.processes.append(p)
    
    def __del__(self):
        """Clean up processes and shared memory."""
        try:
            if hasattr(self, 'running'):
                self.running.value = False
            
            if hasattr(self, 'event_ready'):
                self.event_ready.set()  # Wake up processes to check running state
            
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
        
        # Reset completion counter
        with self.sync_lock:
            self.sync_buffer[0] = 0
        
        # Clear processed event and set ready event
        self.event_processed.clear()
        self.event_ready.set()
        
        # Wait for all processes to complete
        n_processes = len(self.channel_batches)
        while self.sync_buffer[0] < n_processes:
            self.event_processed.wait(timeout=0.001)  # Short timeout to prevent deadlock
            if not self.running.value:
                break
        
        # Update chunk index for next chunk
        self.current_chunk.value += 1
        return self.output_buffer 