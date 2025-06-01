# Channels.py
from collections import namedtuple

AnalogChannel = namedtuple('AnalogChannel', ['name', 'min_val', 'max_val'])

DigitalChannel = namedtuple('DigitalChannel', ['name', 'min_val', 'max_val'])