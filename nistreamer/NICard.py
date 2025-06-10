from typing import List, Optional
from nistreamer.Sequences import Sequence, DOSequence, AOSequence
import numpy as np
import os

class NICard:
    """
    Represents a National Instruments card with its metadata and associated sequences.
    This class holds information about the device and manages its analog and digital sequences.
    """

    def __init__(
        self,
        device_name: str,  # e.g., "PXI1Slot3"
        sample_rate: int,
        sequences: Optional[List[Sequence]] = None,
        trigger_source: str = None,
        clock_source: str = None,
        is_primary: bool = False
    ):
        """
        Initialize a NI card with its metadata and sequences.

        Args:
            device_name (str): Name of the NI device (e.g., "PXI1Slot3")
            sample_rate (int): Sample rate in Hz for all sequences
            sequences (List[Sequence], optional): List of sequences
            is_digital (bool, optional): Whether the card is digital. If True, will be used to determine the sample rate for the digital output.
            trigger_source (str)
            is_primary (bool, optional): Whether the card is the primary card. If True, will be used to determine the sample rate for the digital output.
        """
        self.device_name = device_name
        self.sequences = sequences or []

        # Save the sample rates for analog and digital output
        self.sample_rate = int(sample_rate)

        # Whether the card is the primary card
        self.is_primary = is_primary

        # Placeholder for channel IDs
        self.channel_ids = set()

        # Placeholder for the trigger source and clock source
        self.trigger_source = trigger_source
        self.clock_source = clock_source

        # Create shared memory segment names for this card
        self.shm_name = f"nishm_{os.getpid()}_{self.device_name}"

        # Validate the sequences
        self._validate_sequences()

    def _validate_sequences(self):
        # Make sure all the channel IDs from the sequences are unique and print the repeated ones.
        self.channel_ids, counts = np.unique([seq.channel_id for seq in self.sequences], return_counts=True)
        if len(self.channel_ids) != len(self.sequences):
            raise ValueError(f"Reused channels found: {list(self.channel_ids[counts > 1])}")
        
        # Make sure all the sequences have the same sample rate and that they match the card sample rate
        if not np.all(np.array([seq.sample_rate for seq in self.sequences]) == self.sample_rate):
            raise ValueError(f"Sequences do not match the card sample rate")
        
        # If there are sequences, infer whether the card is analog or digital
        if len(self.sequences):
            self.is_digital = isinstance(self.sequences[0], DOSequence)

            # Ensure all sequences are of the same type
            mode_agreement = [isinstance(seq, DOSequence if self.is_digital else AOSequence) for seq in self.sequences]
            if not np.all(mode_agreement):
                raise ValueError(f"Card {self.device_name} is {'digital' if self.is_digital else 'analog'} but some sequences are not ({[self.sequences[i] for i in np.where(~np.array(mode_agreement))[0]]})")

    def add_sequence(self, sequence: Sequence):
        """Add an analog output sequence to the card."""
        # Make sure the sequence has the same sample rate as the card
        if sequence.sample_rate != self.sample_rate:
            raise ValueError(f"Sequence sample rate {sequence.sample_rate} does not match card sample rate {self.sample_rate}")
        
        # Check for duplicate channel IDs
        if sequence.channel_id in self.channel_ids:
            raise ValueError(f"Channel ID {sequence.channel_id} already exists")
        
        # Make sure the sequence is of the same type as the card
        if isinstance(sequence, DOSequence) != self.is_digital:
            raise ValueError(f"Sequence {sequence} is {'digital' if isinstance(sequence, DOSequence) else 'analog'} but card {self.device_name} is {'digital' if self.is_digital else 'analog'}")
        
        self.sequences.append(sequence)
        self.channel_ids.add(sequence.channel_id)

    def compile(self, chunk_size: int, external_stop_time: float = None):
        """
        Compile all sequences with the given chunk size.

        Args:
            chunk_size (int): The chunk size for all sequences
            stop_time (float, optional): The stop time to extend the sequences to. If not provided, align only to the current card sequences.
        """
        # Make sure the chunk sizes are integers
        if not isinstance(chunk_size, int):
            raise ValueError(f"Chunk size must be an integer, got {chunk_size}")

        # Compile the sequences
        if self.sequences:
             # Calculate the max stop sample for analog output
            max_stop_sample = max([max([ins.end_sample for ins in seq.instructions]) for seq in self.sequences])
            max_stop_time = max_stop_sample / self.sample_rate
            if external_stop_time:
                if external_stop_time < max_stop_time:
                    raise ValueError(f"External stop time {external_stop_time} is less than the current card stop time {max_stop_time}")
                max_stop_sample = int(external_stop_time * self.sample_rate)
            num_chunks = int(np.ceil(max_stop_sample / chunk_size))
            self.num_chunks = num_chunks
            for seq in self.sequences:
                seq.compile(chunk_size, num_chunks)
        else:
            self.num_chunks = 0

        # Save chunk sizes to mark that the sequences are compiled
        self.chunk_size = chunk_size

    def num_channels(self):
        return len(self.channel_ids)

    def get_sequences(self):
        """Returns the pointer to the sequences array. Do not modify the array!"""
        # Make sure the sequences are compiled
        if not self.chunk_size:
            raise ValueError(f"Sequences are not compiled. Run NICard.compile() first.")
        
        return self.sequences