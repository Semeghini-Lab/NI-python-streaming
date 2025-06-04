import numpy as np
from Tags import analog_output, digital_output

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
    return amp * np.sin(2 * np.pi / freq * t + phase)

@analog_output
def linramp(t, start, end):
    return start + (end-start) * t

# ====== DIGITAL OUTPUT COMMANDS ======

@digital_output
def on(t):
    """Digital output on."""
    return np.ones_like(t)

@digital_output
def off(t):
    """Digital output off."""
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