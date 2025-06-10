# Experiment.py

from nistreamer.NICard import NICard
from nistreamer.Sequences import AOSequence, DOSequence

class Experiment:
    def __init__(self, inherit_timings_from_primary=True):
        # Flag for whether to inherit timings from the primary card
        self.inherit_timings_from_primary = inherit_timings_from_primary

        # Dictionary of cards by name
        self.cards = {}
        # Primary card object
        self.primary_card = None

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

    def add_ao_channel(self, card_name, channel_id, name=None, default_value=0.0):
        """Add an analog output channel to a card."""
        # Make sure the card exists
        if card_name not in self.cards:
            raise ValueError(f"Card {card_name} not found.")
        
        seq = AOSequence(
            channel_id=channel_id,
            sample_rate=self.cards[card_name].sample_rate,
            channel_name=name if name else f"{card_name}_ao{channel_id}",
            default_value=default_value,
        )
        
        self.cards[card_name].add_sequence(seq)

    def add_do_channel(self, card_name, port_id, line_id, name=None, default_value=0):
        # Make sure the default value is an integer or a boolean
        if not isinstance(default_value, (int, bool)):
            raise ValueError(f"Default value must be an integer or a boolean, got {type(default_value)}.")
        
        # Make sure the card exists
        if card_name not in self.cards:
            raise ValueError(f"Card {card_name} not found.")
        
        seq = DOSequence(
            channel_id=f"{port_id}/{line_id}",
            sample_rate=self.cards[card_name].sample_rate,
            channel_name=name if name else f"{card_name}_do{port_id}_{line_id}",
            default_value=default_value,
        )
        
        self.cards[card_name].add_sequence(seq)

    def get_cards(self):
        """Return a list of all cards, with the primary card first."""
        return [self.primary_card] + [card for card in self.cards.values() if card != self.primary_card]

    def get_channels_per_card(self):
        """Return a dictionary of all channels per card in the experiment."""
        return {card.device_name: card.sequences for card in self.cards.values()}

if __name__ == "__main__":
    # ----- Add cards -----
    TRIG_LINE = "PXI_Trig0"
    REF_CLK_LINE = "PXI_Trig7"

    exp = Experiment(inherit_timings_from_primary=True)

    # Add the primary card
    exp.add_card("PXI1Slot2", sample_rate=0.4e6, primary=True, trigger_source=TRIG_LINE, clock_source=REF_CLK_LINE)

    # ----- Add remaining analog cards -----
    '''
    For NI PXIe-6739 the max sampling rates are:
    - 1-8 channels (1 per bank of 4 consecutive channels): 1e6 Hz max
    - otherwise: 0.4e6 Hz max
    '''
    exp.add_card("PXI1Slot3", sample_rate=0.4e6)

    # ----- Add remaining digital cards -----
    exp.add_card('PXI1Slot7', sample_rate=10e6)

    ### Main 3D MOT coils
    # 0-10 V for 0-100 A
    # from simulation: 1.16 G/(cm A)
    exp.add_ao_channel('PXI1Slot2', 0, name="MOT_COIL")

    ### MOT bias coils
    exp.add_ao_channel('PXI1Slot2', 8, name="MOT_COIL_BIAS_ARM_Z")
    exp.add_ao_channel('PXI1Slot2', 9, name="MOT_COIL_BIAS_ARM_A")
    exp.add_ao_channel('PXI1Slot2', 10,name="MOT_COIL_BIAS_ARM_C")

    ### TTL for Rb MOT AOMs
    # logic level 0 = set output amplitude
    # logic level 1 = output off
    exp.add_do_channel('PXI1Slot7', 0, 0, name="MOT_RB_REPUMP_TTL")
    exp.add_do_channel('PXI1Slot7', 0, 1, name="MOT_RB_3D_COOLING_TTL")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    exp.add_do_channel('PXI1Slot7', 0, 2, name="MOT_RB_2D_COOLING_TTL")
    exp.add_do_channel('PXI1Slot7', 1, 1, name="MOT_RB_PROBE_TTL")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

    ### FM for Rb MOT AOMs
    # FM mode set to -/+ 10 V = -/+ 26.2 MHz
    exp.add_ao_channel('PXI1Slot2', 1, name="MOT_RB_REPUMP_FM")
    exp.add_ao_channel('PXI1Slot2', 2, name="MOT_RB_3D_COOLING_FM")
    exp.add_ao_channel('PXI1Slot2', 3, name="MOT_RB_2D_COOLING_FM")
    exp.add_ao_channel('PXI1Slot2', 16, name="MOT_RB_PROBE_FM")