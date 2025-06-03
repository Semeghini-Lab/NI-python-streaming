import numpy as np
from Tags import analog_output

'''
Analog output commands. All function must take the positional argument t [ndarray]
corresponding to the real time in seconds. They must return a ndarray of the same
length as t.

The const function is mandatory.
'''

@analog_output
def const(t, value):
    return np.full(len(t), value)

@analog_output
def sine(t, freq, amp, phase):
    return amp * np.sin(2 * np.pi / freq * t + phase)

@analog_output
def linramp(t, start, end):
    return start + (end-start) * t