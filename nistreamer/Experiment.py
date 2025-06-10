# Experiment.py

class Experiment:
    def __init__(self, inherit_timings_from_primary=True):
        self.inherit_timings_from_primary=True

    def add_card(self, card_name, sample_rate, is_primary=False, trigger_source=None, clock_source=None):
        pass

    def add_ao_channel(self, card_name, channel_id, name=None):
        pass

    def add_ao_channel(self, card_name, port_id, line_id, name=None):
        pass