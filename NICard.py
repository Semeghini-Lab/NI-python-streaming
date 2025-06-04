from typing import List, Optional
from Sequences import AOSequence, DOSequence
import numpy as np

class NICard:
    """
    Represents a National Instruments card with its metadata and associated sequences.
    This class holds information about the device and manages its analog and digital sequences.
    """

    def __init__(
        self,
        device_name: str,  # e.g., "PXI1Slot3"
        sample_rate_ao: int,
        sample_rate_do: Optional[int] = None,
        sequence_ao: Optional[List[AOSequence]] = None,
        sequence_do: Optional[List[DOSequence]] = None,
        trigger_source: str = None,
        is_master: bool = False
    ):
        """
        Initialize a NI card with its metadata and sequences.

        Args:
            device_name (str): Name of the NI device (e.g., "PXI1Slot3")
            sample_rate_ao (int): Sample rate in Hz for analog output
            sample_rate_do (int, optional): Sample rate in Hz for digital output. If None, will use the sample rate of the analog output.
            sequence_ao (List[AOSequence], optional): List of analog output sequences
            sequence_do (List[DOSequence], optional): List of digital output sequences
            trigger_source (str)
            is_master (bool, optional): Whether the card is the master card. If True, will be used to determine the sample rate for the digital output.
        """
        self.device_name = device_name
        self.sequence_ao = sequence_ao or []
        self.sequence_do = sequence_do or []

        # Save the sample rates for analog and digital output
        self.sample_rate_ao = int(sample_rate_ao)
        
        # Set sample rate for digital output if provided, otherwise use the sample rate of the analog output
        if sample_rate_do:
            self.sample_rate_do = int(sample_rate_do)
        else:
            self.sample_rate_do = self.sample_rate_ao

        self.is_master = is_master

        # Placeholder for the chunk sizes
        self.chunk_size_ao = None
        self.chunk_size_do = None

        # Placeholder for the trigger source
        self.trigger_source = trigger_source

        # Validate the sequences
        self._validate_ao_sequences()
        self._validate_do_sequences()

        # Setup trigger source
        self._setup_trigger()

    def _setup_trigger(self):
        """If master, create the string for the trigger source. If not, just save."""
        if self.is_master:
            if not self.trigger_source:
                raise ValueError(f"Trigger source is not set for master card")
            self.trigger_source = f"{self.device_name}/{self.trigger_source}"
        else:
            self.trigger_source = self.trigger_source


    def _validate_ao_sequences(self):
        # Make sure all the channel IDs from the analog sequences are unique and print the repeated ones.
        self.channel_ids_ao = set([seq.channel_id for seq in self.sequence_ao])
        if len(self.channel_ids_ao) != len(self.sequence_ao):
            raise ValueError(f"Reused analog channels found")
        
        # Make sure all the sequences have the same sample rate and that they match the card sample rate
        if not np.all(np.array([seq.sample_rate for seq in self.sequence_ao]) == self.sample_rate_ao):
            raise ValueError(f"Analog sequences do not match the card sample rate")

    def _validate_do_sequences(self):
        # Make sure all the channel IDs from the digital sequences are unique and print the repeated ones.
        self.channel_ids_do = set([seq.channel_id for seq in self.sequence_do])
        if len(self.channel_ids_do) != len(self.sequence_do):
            raise ValueError(f"Reused digital channels found")
        
        # Make sure all the sequences have the same sample rate and that they match the card sample rate
        if not np.all(np.array([seq.sample_rate for seq in self.sequence_do]) == self.sample_rate_do):
            raise ValueError(f"Digital sequences do not match the card sample rate")

    def add_ao_sequence(self, sequence: AOSequence):
        """Add an analog output sequence to the card."""
        # Make sure the sequence has the same sample rate as the card
        if sequence.sample_rate != self.sample_rate_ao:
            raise ValueError(f"Sequence sample rate {sequence.sample_rate} does not match card sample rate {self.sample_rate_ao}")
        
        # Check for duplicate channel IDs
        if sequence.channel_id in self.channel_ids_ao:
            raise ValueError(f"Channel ID {sequence.channel_id} already exists")
        
        self.sequence_ao.append(sequence)

    def add_do_sequence(self, sequence: DOSequence):
        """Add a digital output sequence to the card."""
        # Make sure the sequence has the same sample rate as the card
        if sequence.sample_rate != self.sample_rate_do:
            raise ValueError(f"Sequence sample rate {sequence.sample_rate} does not match card sample rate {self.sample_rate_do}")
        
        # Check for duplicate channel IDs
        if sequence.channel_id in self.channel_ids_do:
            raise ValueError(f"Channel ID {sequence.channel_id} already exists")
        
        self.sequence_do.append(sequence)

    def compile(self, chunk_size_ao: int, chunk_size_do: int):
        """
        Compile all sequences with the given chunk size.

        Args:
            chunk_size_ao (int): The chunk size for analog output
            chunk_size_do (int): The chunk size for digital output
        """
        # Make sure the chunk sizes are integers
        if not isinstance(chunk_size_ao, int):
            raise ValueError(f"Chunk size for analog output must be an integer, got {chunk_size_ao}")
        if not isinstance(chunk_size_do, int):
            raise ValueError(f"Chunk size for digital output must be an integer, got {chunk_size_do}")
        
        # Calculate the max stop sample for analog output
        max_stop_sample_ao = max([seq.stop_sample for seq in self.sequence_ao])
        num_chunks_ao = np.ceil(max_stop_sample_ao / chunk_size_ao)

        # Calculate the max stop sample for digital output
        max_stop_sample_do = max([seq.stop_sample for seq in self.sequence_do])
        num_chunks_do = np.ceil(max_stop_sample_do / chunk_size_do)

        # Compile the sequences
        for seq in self.sequence_ao:
            seq.compile(chunk_size_ao, num_chunks_ao)
        for seq in self.sequence_do:
            seq.compile(chunk_size_do, num_chunks_do)

        # Save chunk sizes to mark that the sequences are compiled
        self.chunk_size_ao = chunk_size_ao
        self.chunk_size_do = chunk_size_do

    def get_sequences(self):
        # Make sure the sequences are compiled
        if not self.chunk_size_ao or not self.chunk_size_do:
            raise ValueError(f"Sequences are not compiled. Run NICard.compile() first.")
        
        return self.sequence_ao, self.sequence_do
    
    def get_trigger_source(self):
        if self.is_master:
            return self.trigger_source
        else:
            return None