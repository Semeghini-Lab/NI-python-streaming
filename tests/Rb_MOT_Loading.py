#from setup import *
import numpy as np
import os, sys, time
sys.path.append(r"Z:/sequences/helper-library")
#sys.path.append(r"Z:/alvium-camera")
from alvium import AlviumCamera
#from calibration import *  

from nistreamer.Experiment import Experiment

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

def setup_ni():
    TRIG_LINE = "PXI_Trig0"
    REF_CLK_LINE = "PXI_Trig7"

    exp = Experiment(inherit_timings_from_primary=True, add_channels_to_namespace=True)

    # Add the primary card
    exp.add_card("PXI1Slot2", sample_rate=0.4e6, primary=True, trigger_source=TRIG_LINE, clock_source=REF_CLK_LINE)

    # ----- Add remaining analog cards -----
    '''
    For NI PXIe-6739 the max sampling rates are:
    - 1-8 channels (1 per bank of 4 consecutive channels): 1e6 Hz max
    - otherwise: 0.4e6 Hz max
    '''
    #exp.add_card("PXI1Slot3", sample_rate=0.4e6)

    # ----- Add remaining digital cards -----
    exp.add_card(name='PXI1Slot7',sample_rate=10e6)

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

    ### AM for Rb MOT AOMs
    # 0 V corresponds to set amplitude
    # -1 V corresponds to minimum amplitude
    exp.add_ao_channel('PXI1Slot2', 4, name="MOT_RB_REPUMP_AM")
    exp.add_ao_channel('PXI1Slot2', 5, name="MOT_RB_3D_COOLING_AM")
    exp.add_ao_channel('PXI1Slot2', 6, name="MOT_RB_2D_COOLING_AM")

    ### TTL for Yb MOT AOMs
    # logic level 0 = set output amplitude
    # logic level 1 = output off
    exp.add_do_channel('PXI1Slot7', 0, 4, name="MOT_YB_3D_399_TTL")
    exp.add_do_channel('PXI1Slot7', 0, 5, name="MOT_YB_2D_399_TTL")
    exp.add_do_channel('PXI1Slot7', 0, 6, name="MOT_YB_ZEEMAN_399_TTL")
    exp.add_do_channel('PXI1Slot7', 0, 7, name="MOT_YB_3D_556_TTL")
    exp.add_do_channel('PXI1Slot7', 1, 0, name="YB_PROBE_399_TTL") 

    ### FM for Yb MOT AOMs
    # FM mode set to -/+ 10 V = -/+ 26.2 MHz
    exp.add_ao_channel('PXI1Slot2', 7, name="MOT_YB_3D_399_FM")
    exp.add_ao_channel('PXI1Slot2', 14, name="MOT_YB_3D_399_AM")
    exp.add_ao_channel('PXI1Slot2', 15, name="MOT_YB_2D_399_FM")
    exp.add_ao_channel('PXI1Slot2', 18, name="MOT_YB_ZS_399_FM")
    exp.add_ao_channel('PXI1Slot2', 11, name="MOT_YB_3D_556_FM")
    exp.add_ao_channel('PXI1Slot2', 12, name="MOT_YB_3D_556_AM")

    # Camera triggers
    exp.add_do_channel('PXI1Slot7', 0, 3, name="CAM_TRIG_FLUOR")
    exp.add_do_channel('PXI1Slot7', 1, 2, name="CAM_TRIG_ABS")

    return exp


if __name__ == '__main__':

    print("Starting Rb MOT Loading sequence")

    # --- PREAMBLE --- #
    exp = setup_ni()
    cal = setup_calibration()

    # ---------- Define Allied Vision Cameras ----------
    FLUOR_CAM      = 'DEV_1AB22C051780'
    ABS_CAM        = 'DEV_1AB22C060BE3'

    metadata_to_save = {'MOT_COIL': exp.MOT_COIL}
    camera = AlviumCamera(exp, FLUOR_CAM, exp.CAM_TRIG_FLUOR, metadata_to_save)
    # camera = AlviumCamera(exp, ABS_CAM, CAM_TRIG_ABS, metadata_to_save, gain=48)

    # --- VARIABLES --- #

    us = 1e-6
    ms = 1e-3
    MHz = 1e6

    loading_gradient = cal('MOT_COIL', 25.781, inverse=True)
    loading_bias_Z = 9.5
    loading_bias_A = 3
    loading_bias_C = 7
    loading_AM = 0
    loading_FM = cal('MOT_RB_3D_COOLING_FM', -3 * MHz, inverse=True)
    # loading_time = 5000 * ms
    loading_time = np.linspace(0, 19, 20)

    exposure_time = 1 * ms

    resonance_FM = cal("MOT_RB_3D_COOLING_FM", -1.2 * MHz, inverse=True)

    tof = np.linspace(0.1, 20, 20) * ms
    t = 0

    # --- FUNCTIONS --- #

    def load_mot(t, frequency=loading_FM, amplitude=loading_AM):

        # 3D MOT cooling light
        exp.MOT_RB_3D_COOLING_TTL.low(t=t)
        exp.MOT_RB_3D_COOLING_AM.linramp(t=t, duration=1 * ms, start=0, end=amplitude)
        exp.MOT_RB_3D_COOLING_FM.linramp(t=t, duration=1 * ms, start=0, end=frequency)

        # Repump light
        exp.MOT_RB_REPUMP_TTL.low(t=t)
        exp.MOT_RB_REPUMP_AM.linramp(t=t, duration=1 * ms, start=0, end=0)
        exp.MOT_RB_REPUMP_FM.linramp(t=t, duration=1 * ms, start=0, end=0)

        # 2D MOT cooling light
        exp.MOT_RB_2D_COOLING_TTL.low(t=t)
        exp.MOT_RB_2D_COOLING_AM.linramp(t=t, duration=1 * ms, start=0, end=0)
        exp.MOT_RB_2D_COOLING_FM.linramp(t=t, duration=1 * ms, start=0, end=0)

        t += 2 * ms

        return t
    
    # --- MAIN SEQUENCE --- #

    # Turn on main coils
    exp.MOT_COIL.linramp(t=t, duration=10 * ms, start=0, end=loading_gradient) 
    exp.MOT_COIL_BIAS_ARM_Z.linramp(t=t, duration=1 * ms, start=0, end=loading_bias_Z)
    exp.MOT_COIL_BIAS_ARM_A.linramp(t=t, duration=1 * ms, start=0, end=loading_bias_A)
    exp.MOT_COIL_BIAS_ARM_C.linramp(t=t, duration=1 * ms, start=0, end=loading_bias_C)

    # Turn off MOT for 100 ms
    exp.MOT_RB_3D_COOLING_TTL.high(t=t)
    exp.MOT_RB_PROBE_TTL.high(t=t)
    t += 100 * ms

    # Load MOT
    t = load_mot(t)

    # Imaging
    for _ in range(len(loading_time)): 
        camera.trig(time=t, exp_time=exposure_time)
        t += 1000 * ms


    exp.compile()
    streamer = exp.create_streamer(
        num_workers=2,
        num_writers=2,
        pool_size=8
    )

    seq_path=os.path.abspath(os.path.join(os.path.dirname(__file__), 'Rb_MOT_Loading'))
    seq_name='test'
    camera.start(save_to=f'{seq_path}', launch_live_analysis=True, seq_name=seq_name, num_reps=5)
    streamer.start()
    #run_sequence(seq_name=seq_name, nreps=1) 
    camera.stop()
    print(seq_name)