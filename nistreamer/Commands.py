import numpy as np
from nistreamer.Tags import *

'''
Analog output commands. All function must take the positional argument t [ndarray]
corresponding to the real time in seconds. They must return a ndarray of the same
length as t.

The const function is mandatory.
'''

# ====== ANALOG OUTPUT COMMANDS ======

@analog_output
def const(t, value):
    return np.full_like(t, value)

@analog_output
def sine(t, freq, amp, phase):
    return amp * np.sin(2 * np.pi * freq * t + phase)

@analog_output
@propagate_duration
def linramp(t, start, end, cmd_duration=None):
    return start + (end-start) * t / cmd_duration

@analog_output
@propagate_duration
@propagate_start_value
def rampto(t, value, start=None, cmd_duration=None):
    return start + (value-start) * t / cmd_duration

# ====== DIGITAL OUTPUT COMMANDS ======

@digital_output
@instantaneous
def high(t):
    """Digital output high."""
    return np.ones_like(t)

@digital_output
@instantaneous
def low(t):
    """Digital output low."""
    return np.zeros_like(t)

@digital_output
def square(t, freq, duty_cycle=0.5, phase=0):
    """Generate a square wave.
    
    Args:
        t: Time array
        freq: Frequency in Hz
        duty_cycle: Duty cycle (0-1)
        phase: Phase offset in seconds
    """
    period = 1.0 / freq
    t_phase = (t + phase) % period
    return np.where(t_phase < period * duty_cycle, 1, 0)