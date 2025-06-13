# Experiment.py

import time

from nistreamer.NICard import NICard
from nistreamer.Sequences import AOSequence, DOSequence
from nistreamer.SequenceStreamer import SequenceStreamer

class Experiment:
    def __init__(self, inherit_timings_from_primary=True, add_channels_to_namespace=True):
        # Flag for whether to inherit timings from the primary card
        self.inherit_timings_from_primary = inherit_timings_from_primary

        # Flag for whether to add channels to the namespace
        self.add_channels_to_namespace = add_channels_to_namespace

        # Dictionary of cards by name
        self.cards = {}
        # Primary card object
        self.primary_card = None

        # Compile variables
        self.total_time = None
        self.is_compiled = False

        # Streamer variables, not set by default
        self.num_workers = None
        self.num_writers = None
        self.pool_size = None

    def add_card(self, name, sample_rate, primary=False, trigger_source=None, clock_source=None):
        """Creates and adds a new NICard at a given sample rate.

        Args:
            name (str): The name of the card.
            sample_rate (float): The sample rate of the card.
            is_primary (bool): Whether the card is the primary card.
            trigger_source (str): The source of the trigger.
            clock_source (str): The source of the clock.
        """
        # Check that if the primary card exists, it is not being set again
        if primary and self.primary_card:
            raise ValueError(f"Primary card already set: {self.primary_card}")

        # If the card is primary
        if primary:
            if trigger_source:
                print(f"Primary card {name} will export trigger to {trigger_source}.")
            if clock_source:
                print(f"Primary card {name} will export 10 MHz reference clock to {clock_source}.")

            # Create the card
            self.primary_card = NICard(
                device_name=name,
                sample_rate=sample_rate,
                is_primary=True,
                trigger_source=trigger_source,
                clock_source=clock_source,
            )

            # Add the card to the dictionary
            self.cards[name] = self.primary_card
        # If the card is not primary
        else:
            if self.inherit_timings_from_primary:
                if self.primary_card is None:
                    raise ValueError("Primary card not set! Either add it first or disable trigger and clock inheritance.")
                else:
                    print(f"Card {name} will inherit trigger ({self.primary_card.trigger_source}) and clock ({self.primary_card.clock_source}) from the primary card {self.primary_card.device_name}.")
                    card = NICard(
                        device_name=name,
                        sample_rate=sample_rate,
                        is_primary=False,
                        trigger_source=self.primary_card.trigger_source,
                        clock_source=self.primary_card.clock_source,
                    )
                    self.cards[name] = card
            else:
                """TODO: add support for non-primary cards with no trigger or clock source"""
                pass

    def add_ao_channel(self, card_name: str, channel_id: int, name: str = None, default_value: float = 0.0):
        """Add an analog output channel to a card."""
        # Make sure the card exists
        if card_name not in self.cards:
            raise ValueError(f"Card {card_name} not found.")
        
        # Make sure the channel id is an integer
        if not isinstance(channel_id, int):
            raise ValueError(f"Channel id must be an integer, got {type(channel_id)}.")
        
        # Create the sequence
        seq = AOSequence(
            channel_id=f"ao{channel_id}",
            sample_rate=self.cards[card_name].sample_rate,
            channel_name=name if name else f"{card_name}_ao{channel_id}",
            default_value=default_value,
        )

        # Undo the compilation status if a new sequence is added
        if self.is_compiled:    
            print("Warning: adding a new sequence after compilation will revoke the compilation status.")
            self.is_compiled = False

        # Add channel to namespace
        if self.add_channels_to_namespace:
            if hasattr(self, name):
                raise ValueError(f"Channel {name} already exists in the namespace.")
            else:
                setattr(self, name, seq)
        
        self.cards[card_name].add_sequence(seq)

        return seq

    def add_do_channel(self, card_name: str, port_id: int, line_id: int, name: str = None, default_value: int = 0):
        # Make sure the default value is an integer or a boolean
        if not isinstance(default_value, (int, bool)):
            raise ValueError(f"Default value must be an integer or a boolean, got {type(default_value)}.")
        
        # Make sure the port and line ids are integers
        if not isinstance(port_id, int) or not isinstance(line_id, int):
            raise ValueError(f"Port and line ids must be integers, got {type(port_id)} and {type(line_id)}.")
        
        # Make sure the card exists
        if card_name not in self.cards:
            raise ValueError(f"Card {card_name} not found.")
        
        # Create the sequence
        seq = DOSequence(
            channel_id=f"port{port_id}/line{line_id}",
            sample_rate=self.cards[card_name].sample_rate,
            channel_name=name if name else f"{card_name}_do{port_id}_{line_id}",
            default_value=default_value,
        )

        # Undo the compilation status if a new sequence is added
        if self.is_compiled:
            print("Warning: adding a new sequence after compilation will revoke the compilation status.")
            self.is_compiled = False

        # Add channel to namespace
        if self.add_channels_to_namespace:
            if hasattr(self, name):
                raise ValueError(f"Channel {name} already exists in the namespace.")
            else:
                setattr(self, name, seq)
        
        self.cards[card_name].add_sequence(seq)

        return seq

    def compile(self, chunk_size: int = 65536, total_time: float = None):
        """Compile the experiment."""
        
        # If there is a primary card, make sure it has at least one channel defined
        if self.primary_card:
            if self.primary_card.sequences is None:
                raise ValueError("Primary card has no sequences.")
            
        # Start the timing of the compilation
        compile_time = time.time()
            
        # If the total time is not specified, find the longest sequence
        if total_time is None:
            # Find the instruction that ends last
            total_time = max([max([max(seq.instructions, key=lambda x: x.end_sample).end_sample if seq.instructions else 0 for seq in card.sequences])/card.sample_rate if card.sequences else 0 for card in self.cards.values()])
        
        if total_time == 0.0:
            raise NotImplementedError("Total time of the experiment is 0.0, which is not supported.")
        
        print(f"Compiling experiment...", end="")

        # Compile all cards
        for card in self.cards.values():
            card.compile(chunk_size=chunk_size, external_stop_time=total_time)

        print(f" done in {(time.time() - compile_time)*1e3:.3f} ms.")

        self.is_compiled = True
        return total_time

    def create_streamer(self, num_workers=None, num_writers=None, pool_size=None):
        """Create a streamer object."""
        if num_workers is None:
            num_workers = self.num_workers
        if num_writers is None:
            num_writers = self.num_writers
        if pool_size is None:
            pool_size = self.pool_size
        
        # Check that necessary variables are set
        if num_workers is None or num_writers is None or pool_size is None:
            raise ValueError("Number of workers, writers, and pool size must be set.")
        
        # Make sure the experiment is compiled
        if not self.is_compiled:
            raise ValueError("Experiment is not compiled. Please compile the experiment before creating a streamer.")
        
        # Create the streamer
        streamer = SequenceStreamer(
            num_workers=num_workers,
            num_writers=num_writers,
            pool_size=pool_size,
            cards=self.get_cards(),
        )
        return streamer

    def run(self):
        streamer = self.create_streamer()
        streamer.run()

    def get_cards(self):
        """Return a list of all cards, with the primary card first."""
        return [self.primary_card] + [card for card in self.cards.values() if card != self.primary_card]

    def get_channels_per_card(self):
        """Return a dictionary of all channels per card in the experiment."""
        return {card.device_name: card.sequences for card in self.cards.values()}

if __name__ == "__main__":
    # Variables:
    ms = 1e-3


    # ----- Add cards -----
    TRIG_LINE = "PXI_Trig0"
    REF_CLK_LINE = "PXI_Trig7"

    exp = Experiment(inherit_timings_from_primary=True)

    # Add the primary card
    exp.add_card("PXI1Slot2", sample_rate=0.4e6, primary=True, trigger_source=TRIG_LINE, clock_source=REF_CLK_LINE)

    # ----- Add remaining digital cards -----
    exp.add_card('PXI1Slot7', sample_rate=10e6)

    ### Main 3D MOT coils
    # 0-10 V for 0-100 A
    # from simulation: 1.16 G/(cm A)
    mot_coil = exp.add_ao_channel('PXI1Slot2', 0, name="MOT_COIL")

    ### TTL for Rb MOT AOMs
    # logic level 0 = set output amplitude
    # logic level 1 = output off
    exp.add_do_channel('PXI1Slot7', 0, 0, name="MOT_RB_REPUMP_TTL")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

    ### FM for Rb MOT AOMs
    # FM mode set to -/+ 10 V = -/+ 26.2 MHz
    exp.add_ao_channel('PXI1Slot2', 1, name="MOT_RB_REPUMP_FM")

    mot_coil.linramp(0, 1 * ms, start=0, end=10)
    mot_coil.sine(1*ms, cmd_duration=1 * ms, freq=1.0, amp=10, phase=0)

    exp.compile(chunk_size=65536)