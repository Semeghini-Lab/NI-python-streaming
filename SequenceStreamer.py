# SequenceStreamer.py
# Created by: Marcin Kalinowski & Yi Zhu

import sys
import struct
import numpy as np
import zmq

import heapq

from Sequences import AOSequence, DOSequence
from NICard import NICard
from Worker import Worker
from Writer import Writer   

import multiprocessing

if sys.platform == "darwin":
    multiprocessing.set_start_method("fork")


class SequenceStreamer:
    """
    Manager class: sets up shared memory, manages worker and writer processes,
    and runs the even loop in the main thread to assign chunks and collect reports.
    """

    WORKER_ASSIGN_STRUCT = struct.Struct('>q i I I I')  # (chunk_idx, buf_idx, card_idx, ch_start, ch_end)
    WORKER_DONE_STRUCT = struct.Struct('>q i I I d')  # (chunk_idx, buf_idx, card_idx, worker_id, compute_time)
    WRITER_ASSIGN_STRUCT = struct.Struct('>q i I')  # (chunk_idx, buf_idx, card_idx)
    WRITER_DONE_STRUCT = struct.Struct('>q i I')  # (chunk_idx, buf_idx, card_idx)

    def __init__(
            self,
            cards: list[NICard],  # List of cards to stream to
            num_workers: int,  # Number of worker processes
            num_writers: int,  # Number of writer processes
            pool_size: int,  # Size of the memory pool
    ) -> None:
        self.cards = cards
        self.num_workers = num_workers
        self.num_writers = num_writers
        self.pool_size = pool_size

        # Validate the vards
        self._validate_cards()

        # Get the card sequences (also checks compile status)
        self.sequences = [card.get_sequences() for card in self.cards]

        # Calculate the number of chunks to stream for each card
        self.num_chunks_to_stream = [int(card.sequences[0].instructions[-1].end_sample // card.chunk_size) if card.sequences else 0 for card in self.cards]

        # Create shared memory segments for the data buffers
        self.shms = []
        self._initialize_shared_memory()

        # Initialize ZMQ context
        self.context = zmq.Context()

        # Initialize ZMQ sockets for writer communication
        self.writer_ready_sockets = [] # PUSH sockets for sending ready notifications
        self.writer_report_sockets = [] # PULL sockets for receiving report notifications

        # Initialize ZMQ sockets for worker communication
        self.worker_assign_sockets = []  # PUSH sockets for sending assignments
        self.worker_done_sockets = []    # PULL sockets for receiving completion reports

        # Initialize a callback map for monitored events
        self.event_callbacks = {}

        # Create poller for dispatching messages
        self.poller = zmq.Poller()

        # Initialize sockets
        self._initialize_sockets()

        # Divide cards between writers
        self.writer_card_groups = []
        self._allocate_cards_to_writers()

        # Divide cards and their channels for worker dispatching
        self.channel_ranges = []
        self.num_channel_ranges = []
        self._allocate_resources_to_workers()

        # Placeholder for processes
        self.workers = []
        self.writers = []

        # Number of chunks to stream
        self.num_chunks_to_stream = [card.num_chunks for card in self.cards]
        self.total_num_chunks_to_stream = sum(self.num_chunks_to_stream)

        # Prepare the order of chunks to stream
        self.chunk_queue = []
        self._precompute_chunk_queue()


        assert len(self.chunk_queue) == np.sum(np.array(self.num_chunks_to_stream)*np.array(self.num_channel_ranges)), "Precomputed chunk queue does not match the total number of chunks to stream."

        # Manager internal state
        self.available_workers = [] # Filled in at startup
        self.available_slots = [list(range(self.pool_size)) for _ in range(len(self.cards))]
        self.chunks_being_processed = [{} for _ in range(len(self.cards))] # Keep track currently processed chunks (chunk_idx -> buf_idx, remaining_channel_groups)
        self.chunks_completed = [[] for _ in range(len(self.cards))] # Keep track of which chunks are completed for each card
        self.next_chunk_to_write = [0 for _ in range(len(self.cards))] # Keep track of the next chunk to write for each card

    def _validate_cards(self):
        """
        Validate the cards, make sure the primary card is first.
        """
        # Make sure the cards are compiled
        for card in self.cards:
            if not card.num_chunks:
                raise ValueError(f"Card {card.device_name} is not compiled. Run NICard.compile() first.")

        # Keep only the cards whose sequences have more than 0 samples
        self.cards = [card for card in self.cards if card.num_chunks > 0]

        # If there is a primary card, make sure there is only one and place it first
        if any([card.is_primary for card in self.cards]):
            if sum([card.is_primary for card in self.cards]) > 1:
                raise ValueError("Multiple primary cards found.")
            self.cards = [card for card in self.cards if card.is_primary] + [card for card in self.cards if not card.is_primary]
        else:
            print("Warning: No primary card found, synchronization is not guaranteed.")

    def _get_port(self, socket):
        """
        Get the port of a socket.
        """
        return int(socket.getsockopt(zmq.LAST_ENDPOINT).decode().split(":")[-1])

    def _initialize_sockets(self):
        """
        Initialize the ZMQ sockets.
        """
        for i in range(len(self.cards)):
            # Create the writer ready and report sockets
            self.writer_ready_sockets.append(self.context.socket(zmq.PUSH))
            self.writer_report_sockets.append(self.context.socket(zmq.PULL))

            # Bind the writer ready socket to a random port
            self.writer_ready_sockets[i].bind_to_random_port(f"tcp://127.0.0.1")
            self.writer_report_sockets[i].bind_to_random_port(f"tcp://127.0.0.1")

        for i in range(self.num_workers):
            # Create the worker assign and done sockets
            self.worker_assign_sockets.append(self.context.socket(zmq.PUSH))
            self.worker_done_sockets.append(self.context.socket(zmq.PULL))
            
            # Bind the worker assign socket to a random port
            self.worker_assign_sockets[i].bind_to_random_port(f"tcp://127.0.0.1")
            self.worker_done_sockets[i].bind_to_random_port(f"tcp://127.0.0.1")

    def _allocate_cards_to_writers(self):
        """
        Divide cards between writers.
        """
        cards_per_writer = len(self.cards) // self.num_writers
        card_groups = []
        for i in range(self.num_writers-1):
            card_groups.append(self.cards[i*cards_per_writer:(i+1)*cards_per_writer])
        card_groups.append(self.cards[(self.num_writers-1)*cards_per_writer:])
        self.writer_card_groups = card_groups

    def _allocate_resources_to_workers(self):
        """
        Decide how to divide the channels and cards into groups.
        """
        # Calculate the total number of analog and digital channels across all cards
        total_channels = sum([card.num_channels() for card in self.cards])

        # Calculate the number of channels per worker
        channels_per_worker = int(np.ceil(total_channels / self.num_workers))

        print(f"SequenceStreamer: Allocating {channels_per_worker} channels per worker.")

        # For each card, divide the channels into correct-sized groups
        card_channel_ranges = []
        for card in self.cards:
            # Divide the analog channels into groups
            channel_ranges = []
            if channels_per_worker:
                for i in range(0, card.num_channels(), channels_per_worker):
                    channel_ranges.append((i,min(i+channels_per_worker, card.num_channels())))
            card_channel_ranges.append(channel_ranges)

        self.channel_ranges = card_channel_ranges
        self.num_channel_ranges = [len(channel_ranges) for channel_ranges in card_channel_ranges]

    def _initialize_shared_memory(self):
        """
        Initialize the shared memory segments for the data buffers.
        """
        # Iterate the cards 
        for card in self.cards:
            shm_size = self.pool_size * card.num_channels() * card.chunk_size * np.dtype(bool if card.is_digital else np.float64).itemsize

            # Create shared memory segment
            self.shms.append(multiprocessing.shared_memory.SharedMemory(create=True, size=shm_size, name=card.shm_name))

            print(f"Shared memory segment '{card.shm_name}' created with size {shm_size} bytes: ({self.pool_size}x {card.chunk_size} samples for {card.num_channels()} channels).")

    def _precompute_chunk_queue(self):
        """
        Precompute the queue of chunks to stream based on the end time of the chunks.
        """
        queue = []
        for card_idx, card in enumerate(self.cards):
            for chunk_idx in range(card.num_chunks):
                queue.append((chunk_idx, card_idx))

        # Sort the queue by time
        queue.sort(key=lambda x: -(x[0]+1)/self.cards[x[1]].sample_rate)

        # For each queue element, extend it by the number of channel groups
        queue = [[(chunk_idx, channel_group_idx, card_idx) for channel_group_idx in range(self.num_channel_ranges[card_idx])] for (chunk_idx, card_idx) in queue]

        # Flatten the queue
        queue = [item for sublist in queue for item in sublist]

        # Add the chunks to the queue
        self.chunk_queue = queue

    def start(self):
        try:
            # Spawn worker processes
            for wid in range(self.num_workers):
                # Register worker done socket with poller
                self.poller.register(self.worker_done_sockets[wid], zmq.POLLIN)
                self.event_callbacks[self.worker_done_sockets[wid]] = self._process_worker_done_data

                worker = Worker(
                    worker_id=wid,
                    cards=self.cards,
                    pool_size=self.pool_size,
                    assign_port=self._get_port(self.worker_assign_sockets[wid]),
                    done_port=self._get_port(self.worker_done_sockets[wid]),
                )
                worker.daemon = True
                worker.start()

                # Add as available worker
                self.available_workers.append(wid)

            # Register writer sockets with poller
            for socket in self.writer_report_sockets:
                self.poller.register(socket, zmq.POLLIN)
                self.event_callbacks[socket] = self._process_slot_free

            # Spawn writer processes
            for wid in range(self.num_writers):
                card_indices = [self.cards.index(card) for card in self.writer_card_groups[wid]]
                writer = Writer(
                    writer_id=wid,
                    cards=self.writer_card_groups[wid],
                    card_indices=card_indices,
                    outbuf_num_chunks=self.pool_size,
                    pool_size=self.pool_size,
                    ready_ports=[self._get_port(self.writer_ready_sockets[i]) for i in card_indices],
                    report_ports=[self._get_port(self.writer_report_sockets[i]) for i in card_indices],
                )
                writer.daemon = True
                writer.start()

            # Start the manager loop
            self._manager_loop()

        except Exception as e:
            print(f"Error in start: {e}")
            self.cleanup()
            raise

    def _assign_chunk(self, worker_id: int, chunk_idx: int, buf_idx: int, card_idx: int, ch_start: int, ch_end: int):
        """Assign a chunk to a worker."""
        assign_data = self.WORKER_ASSIGN_STRUCT.pack(chunk_idx, buf_idx, card_idx, ch_start, ch_end)
        self.worker_assign_sockets[worker_id].send(assign_data)
        #print(f"[ASSIGN CALC] card={card_idx} chunk={chunk_idx} slot={buf_idx} worker_id={worker_id} channels={ch_start}-{ch_end}.")

    def _process_worker_done_data(self, socket):
        """Process worker done data."""
        data = socket.recv()
        chunk_idx, buf_idx, card_idx, worker_id, compute_time = self.WORKER_DONE_STRUCT.unpack(data)

        #print(f"[FINISHED CALC] card={card_idx} chunk={chunk_idx} slot={buf_idx} worker_id={worker_id}.")

        if chunk_idx not in self.chunks_being_processed[card_idx]:
            raise ValueError(f"Received done notification for an unexpected chunk {chunk_idx} at card {card_idx}.")
        
        if self.chunks_being_processed[card_idx][chunk_idx][0] != buf_idx:
            raise ValueError(f"Received done notification for an unexpected slot {buf_idx} for chunk {chunk_idx} at card {card_idx}.")        
        
        # Free the worker
        self.available_workers.append(worker_id)

        # Decrement the number of remaining channel groups for this chunk
        self.chunks_being_processed[card_idx][chunk_idx][1] -= 1

        # All channel groups for this chunk have been processed
        if self.chunks_being_processed[card_idx][chunk_idx][1] == 0:
            #print(f"All channel groups for chunk {chunk_idx} at card {card_idx} have been processed.")
            # Remove the chunk from processed chunks
            del self.chunks_being_processed[card_idx][chunk_idx]

            # If the next chunk to write is the same as the current chunk, write it
            if self.next_chunk_to_write[card_idx] == chunk_idx:
                self._assign_slot_write(chunk_idx, buf_idx, card_idx)
            else:
                # Otherwise, queue it for writing
                heapq.heappush(self.chunks_completed[card_idx], (chunk_idx, buf_idx, card_idx))

    def _process_slot_free(self, socket):
        """
        Process a slot free notification from the writer.
        This method reads the slot index from the pipe and adds it as available.
        """
        data = socket.recv()
        chunk_idx, buf_idx, card_idx = self.WRITER_ASSIGN_STRUCT.unpack(data)
        
        if chunk_idx == -1:
            print("Received error message from writer, stopping manager loop.")
            self.running = False
            return
        
        #print(f"[DONE WRITE] card={card_idx} chunk={chunk_idx} slot={buf_idx}.")

        # Add the slot to the available slot list
        self.available_slots[card_idx].append(buf_idx)

        # Increment the number of chunks written
        self.num_chunks_written += 1

        # If it was the last chunk, stop the manager loop
        if self.num_chunks_written == self.total_num_chunks_to_stream:
            self.running = False
            print("Last chunk streamed, stopping manager loop.")

    def _assign_slot_write(self, chunk_idx: int, buf_idx: int, card_idx: int):
        """
        Notify the writer that a chunk is ready to be written.
        This method writes the chunk index and buffer index to the pipe.
        """
        #print(f"[WRITE] card={card_idx} chunk={chunk_idx} slot={buf_idx}.")
        self.writer_ready_sockets[card_idx].send(self.WRITER_ASSIGN_STRUCT.pack(chunk_idx, buf_idx, card_idx))

        if chunk_idx != self.next_chunk_to_write[card_idx]:
            raise ValueError(f"Chunk {chunk_idx} at card {card_idx} is not the next chunk to write.")
        
        self.next_chunk_to_write[card_idx] += 1

    def _manager_loop(self):
        import time
        print(f"Starting manager loop with {self.total_num_chunks_to_stream} chunks to stream: {self.num_chunks_to_stream}.")
        # Start the manager loop
        self.running = True
        self.num_chunks_written = 0
        while self.running:
            # While there are free workers and available slots, assign chunks to be computed until done
            temp_chunk_queue = []
            # Ensure that we loop at most self.available_workers times
            loop_count = 0 
            while self.chunk_queue and len(self.available_workers):# and loop_count < self.num_workers:
                # Get the next highest-priority chunk to stream
                (chunk_idx, channel_group_idx, card_idx) = self.chunk_queue.pop()

                # Check if the chunk is already being assigned to a buffer slot
                if chunk_idx in self.chunks_being_processed[card_idx]:
                    buf_idx = self.chunks_being_processed[card_idx][chunk_idx][0]
                else:
                    # If the chunk is not processed, get the next available slot (1 slot always reserved for the next-in-order chunk)
                    if (len(self.available_slots[card_idx]) > 1) or (len(self.available_slots[card_idx]) and chunk_idx == self.next_chunk_to_write[card_idx]):
                        buf_idx = self.available_slots[card_idx].pop()
                        self.chunks_being_processed[card_idx][chunk_idx] = [buf_idx, self.num_channel_ranges[card_idx]]
                    else:
                        # Put chunk into the temporary queue
                        temp_chunk_queue.append((chunk_idx, channel_group_idx, card_idx))
                        loop_count += 1
                        continue

                # Get the next available worker
                worker_id = self.available_workers.pop()

                # Determine the channel range for this worker
                ch_start, ch_end = self.channel_ranges[card_idx][channel_group_idx]

                # Assign the chunk to the worker
                self._assign_chunk(worker_id, chunk_idx, buf_idx, card_idx, ch_start, ch_end)

                # Increment the loop count
                loop_count += 1

            # Write all in-order chunks from the completed queue
            for card_idx in range(len(self.cards)):
                while self.chunks_completed[card_idx] and self.chunks_completed[card_idx][0][0] == self.next_chunk_to_write[card_idx]:
                    (chunk_idx, buf_idx, card_idx) = heapq.heappop(self.chunks_completed[card_idx])
                    self._assign_slot_write(chunk_idx, buf_idx, card_idx)
            
            # Empty the temporary queue
            self.chunk_queue.extend(reversed(temp_chunk_queue))
            
            # Process events
            for socket, _ in self.poller.poll(0):
                self.event_callbacks[socket](socket)

        # Send messages to stop the writer and workers
        for socket in self.writer_ready_sockets:    
            socket.send(self.WRITER_ASSIGN_STRUCT.pack(-1, -1, 0))

        for socket in self.worker_assign_sockets:
            socket.send(self.WORKER_ASSIGN_STRUCT.pack(-1, -1, 0, 0, 0))

        # Wait for the full buffer to play (2x for safety)
        time.sleep(2 * self.pool_size * max([card.chunk_size/card.sample_rate for card in self.cards])) # in seconds

        # Wait for the writer and workers to finish
        for writer in self.writers:
            writer.join(timeout=10.0) # in seconds
        for worker in self.workers:
            worker.join(timeout=10.0) # in seconds
        print("Sequence completed.")

    def cleanup(self):
        """Clean up all resources."""
        try:
            try:
                for shm in self.shms:
                    shm.close()
                    shm.unlink()
            except FileNotFoundError:
                pass

            # Close ZMQ sockets and context
            for socket in self.writer_ready_sockets:
                socket.close()
            for socket in self.writer_report_sockets:
                socket.close()
            for socket in self.worker_assign_sockets:
                socket.close()
            for socket in self.worker_done_sockets:
                socket.close()
            self.context.term()

        except Exception as e:
            print(f"Error during cleanup: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        self.cleanup()


        
if __name__ == "__main__":

    # Get the sample 
    # rate
    sample_rate = 1_000_000 # Hz

    # Channel 0
    ch0 = AOSequence(channel_id="ao0", sample_rate=sample_rate)
    #ch0.linramp(0.0, 1.0, start=0, end=2)
    #ch0.const(1.0, 1.0, value=5.0)
    #ch0.linramp(3.0, 1.0, start=0, end=6.8)
    ch0.sine(0.0, 10*60.0, freq=1_000, amp=2, phase=0)

    # Channel 1
    ch1 = AOSequence(channel_id="ao0", sample_rate=sample_rate)
    # ch1.const(0.0, 0.5, value=3.0)
    # ch1.linramp(0.5, 0.5, start=3.0, end=0)
    # ch1.sine(1.0, 5, freq=2, amp=2, phase=0)
    # ch1.linramp(6.0, 1.0, start=0, end=-2)
    for i in range(1_000*60):
        ch1.const(0.001*i,0.001, value=i % 2)
    #ch1.sine(0.0, 10*60.0, freq=10_000, amp=2, phase=np.pi/2)

    # Channel 1 on a different card
    ch2 = AOSequence(channel_id="ao1", sample_rate=sample_rate)
    ch2.const(0.0, 0.5, value=3.0)
    ch2.linramp(0.5, 0.5, start=3.0, end=0)
    ch2.sine(1.0, 5, freq=2, amp=2, phase=0)
    ch2.linramp(6.0, 1.0, start=0, end=-2)

    # Channel 1 digital 
    ch3 = DOSequence(channel_id="port2/line0", sample_rate=int(10e6))
    ch3.high(0, 2.0)
    ch3.low(2.0, 0.5)

    # Set the chunk size and compile 
    chunk_size = 65536

    # Create the NICARD object
    card1 = NICard(
        device_name="PXI1Slot3", 
        sample_rate=sample_rate,
        sequences=[ch0],
        is_primary = True,
        trigger_source="PXI_Trig0",
        clock_source="PXI_Trig7"
    )

    card2 = NICard(
        device_name="PXI1Slot4", 
        sample_rate=sample_rate,
        sequences=[ch1],
        trigger_source=card1.trigger_source,
        clock_source=card1.clock_source
    )

    card3 = NICard(
        device_name="PXI1Slot7", 
        sample_rate=int(10e6),
        sequences=[ch3],
        trigger_source=card1.trigger_source,
        clock_source=card1.clock_source
    )

    # Aggregate
    cards = [card1]

    # Compile the cards
    for card in cards:
        card.compile(chunk_size=chunk_size, external_stop_time=10*60.0)

    # Example usage with context manager
    with SequenceStreamer(
        cards=cards, # List of cards to stream to
        num_workers=1, # Number of worker processes
        num_writers=1, # Number of writer processes
        pool_size=8,   # Size of the memory pool
    ) as streamer:
        streamer.start()

