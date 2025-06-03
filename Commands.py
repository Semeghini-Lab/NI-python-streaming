import numpy as np

'''
Analog output commands. All function must take the positional argument t [ndarray]
corresponding to the real time in seconds. They must return a ndarray of the same
length as t.

The const function is mandatory.
'''
def const(t, value):
    return np.full(len(t), value)

def sine(t, freq, amp, phase):
    return amp * np.sin(2 * np.pi / freq * t + phase)

def linramp(t, start, end):
    return start + (end-start) * t