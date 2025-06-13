import numpy as np

def setup_calibration():
    '''
    This file contains calibration functions for converting NI values to physical values and vice versa.

    NOTE: the channel names used here should be identical to the channel names used in setup.py

    All of the calibrations area stored in a dictionary:
    dictionary {(channel, 'FORWARD'): func,
                (channel, 'INVERSE'): func,
                (channel, 'UNITS'): str,
                ...}
    '''

    calibration = {}

    def _cal(channel, value, inverse=False):
        '''
        Converts an NI value to a physical value. If inverse=True, converts a physical value to an NI value.
        '''
        key = (channel, 'FORWARD') if not inverse else (channel, 'INVERSE')
        if key not in calibration:
            raise ValueError(f'Calibration for channel {channel} not defined')
        return calibration[key](value)
    
    cal = np.vectorize(_cal, excluded=['channel', 'inverse'])

    def cal_units(channel):
        '''
        Returns the units of the physical values for a given channel
        '''
        key = (channel, 'UNITS')
        if key not in calibration:
            raise ValueError(f'Calibration for channel {channel} not defined')
        return calibration[key]

    # region ---------- Define calibrations here ----------

    # 3D MOT coil
    # 0-10 V NI for 0-100 A physical
    # from simulation: 1.16 G/(cm A)
    calibration[('MOT_COIL', 'FORWARD')] = lambda x: x*10/2*1.16
    calibration[('MOT_COIL', 'INVERSE')] = lambda x: x/10*2/1.16
    calibration[('MOT_COIL', 'UNITS')] = 'G/cm'

    # 3D MOT bias coils
    # Acopian power supplies: 0-10 V gives 0-7 A
    # from simulation: 1.16 G/(cm A)
    calibration[('MOT_COIL', 'FORWARD')] = lambda x: x*10/2*1.16
    calibration[('MOT_COIL', 'INVERSE')] = lambda x: x/10*2/1.16
    calibration[('MOT_COIL', 'UNITS')] = 'G/cm'

    # 3D cooling detuning from resonance
    calibration[('MOT_RB_3D_COOLING_FM', 'FORWARD')] = lambda x: (150 - 2*(3.05*x+82.5) + 7.58) * 1e6
    calibration[('MOT_RB_3D_COOLING_FM', 'INVERSE')] = lambda x: (((150+7.58)-x/1e6)/2 - 82.5)/3.05
    calibration[('MOT_RB_3D_COOLING_FM', 'UNITS')] = 'Hz'

    # 2D cooling detuning from resonance
    calibration[('MOT_RB_2D_COOLING_FM', 'FORWARD')] = lambda x: (150 - 2*(3.05*x+86.1)) * 1e6
    calibration[('MOT_RB_2D_COOLING_FM', 'INVERSE')] = lambda x: ((150-x/1e6)/2 - 86.1)/3.05
    calibration[('MOT_RB_2D_COOLING_FM', 'UNITS')] = 'Hz'

    # 3D detuning - Yb 399 - assuming same driver slope as Rb cooling
    # AOM center frequency: 105 MHz
    # AOM FM range: 26.2 MHz
    # RF power: 25.1 dBm
    calibration[('MOT_YB_3D_399_FM', 'FORWARD')] = lambda x: (-260 + 2*(3.05*x+105)) * 1e6
    calibration[('MOT_YB_3D_399_FM', 'INVERSE')] = lambda x: ((260+x/1e6)/2 - 105)/3.05
    calibration[('MOT_YB_3D_399_FM', 'UNITS')] = 'Hz (detuning wrt atomic resonance)'

    # 2D detuning - Yb 399 - assuming same driver slope as Rb cooling
    calibration[('MOT_YB_2D_399_FM', 'FORWARD')] = lambda x: (-260 + 2*(3.05*x+110)) * 1e6
    calibration[('MOT_YB_2D_399_FM', 'INVERSE')] = lambda x: ((260+x/1e6)/2 - 110)/3.05
    calibration[('MOT_YB_2D_399_FM', 'UNITS')] = 'Hz (detuning wrt atomic resonance)'

    # AOM center frequency: 78.39 MHz
    # AOM FM range: 0.65536 MHz / V
    # calibration[('MOT_YB_3D_556_FM', 'FORWARD')] = lambda x: (-206.78 + 2s*(0.65536*x+78.39)) * 1e6
    calibration[('MOT_YB_3D_556_FM', 'FORWARD')] = lambda x: (x - 3.09) * 2 * 0.65536e6
    calibration[('MOT_YB_3D_556_FM', 'INVERSE')] = lambda x: x / 0.65536e6 / 2 + 3.09
    calibration[('MOT_YB_3D_556_FM', 'UNITS')] = 'Hz (detuning wrt atomic resonance)'

    return cal
