# Example sequence from 06/10/2025 for an Yb MOT time-of-flight experiment.
# Used for debugging during the transition period to the new python backend.

import numpy as np
import os, sys, time
sys.path.append(r"Z:/sequences/helper-library")
sys.path.append(r"Z:/alvium-camera")
from alvium import AlviumCamera
#from calibration import *  

from nistreamer.Experiment import Experiment

def setup():
    TRIG_LINE = "PXI_Trig0"
    REF_CLK_LINE = "PXI_Trig7"\

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
    exp.add_do_device(name='PXI1Slot7',samp_rate=10e6)

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
    MOT_RB_REPUMP_AM              = exp.add_ao_channel('PXI1Slot2', 4)
    MOT_RB_3D_COOLING_AM          = exp.add_ao_channel('PXI1Slot2', 5)
    MOT_RB_2D_COOLING_AM          = exp.add_ao_channel('PXI1Slot2', 6)

    ### TTL for Yb MOT AOMs
    # logic level 0 = set output amplitude
    # logic level 1 = output off
    MOT_YB_3D_399_TTL             = exp.add_do_channel('PXI1Slot7', 0, 4)
    MOT_YB_2D_399_TTL             = exp.add_do_channel('PXI1Slot7', 0, 5)
    MOT_YB_ZEEMAN_399_TTL         = exp.add_do_channel('PXI1Slot7', 0, 6)
    MOT_YB_3D_556_TTL             = exp.add_do_channel('PXI1Slot7', 0, 7)
    YB_PROBE_399_TTL              = exp.add_do_channel('PXI1Slot7', 1, 0) 

    ### FM for Yb MOT AOMs
    # FM mode set to -/+ 10 V = -/+ 26.2 MHz
    MOT_YB_3D_399_FM              = exp.add_ao_channel('PXI1Slot2', 7)
    MOT_YB_3D_399_AM              = exp.add_ao_channel('PXI1Slot2', 14)
    MOT_YB_2D_399_FM              = exp.add_ao_channel('PXI1Slot2', 15)
    MOT_YB_ZS_399_FM              = exp.add_ao_channel('PXI1Slot2', 18)
    MOT_YB_3D_556_FM              = exp.add_ao_channel('PXI1Slot2', 11)
    MOT_YB_3D_556_AM              = exp.add_ao_channel('PXI1Slot2', 12)

    CAM_TRIG_FLUOR                = exp.add_do_channel('PXI1Slot7', 0, 3)
    CAM_TRIG_ABS                  = exp.add_do_channel('PXI1Slot7', 1, 2)


    return exp, channels


if __name__ == '__main__':

    # --- PREAMBLE --- #
    exp.clear_edit_cache()
    exp.clear_compile_cache()

    metadata_to_save = {'MOT_COIL': MOT_COIL}
    # camera = AlviumCamera(exp, FLUOR_CAM, CAM_TRIG_FLUOR, metadata_to_save, gain=48)
    # camera = AlviumCamera(exp, ABS_CAM, CAM_TRIG_ABS, metadata_to_save, gain=48)

    # --- VARIABLES --- #

    us = 1e-6
    ms = 1e-3
    kHz = 1e3
    MHz = 1e6

    loading_gradient = cal('MOT_COIL', 18, inverse=True)
    loading_bias_Z = 9.5
    loading_bias_A = 3
    loading_bias_C = 7
    loading_AM_399 = 0
    loading_AM_556 = 0
    loading_FM_399 = cal("MOT_YB_3D_399_FM", -53 * MHz, inverse=True)
    loading_FM_556 = cal("MOT_YB_3D_556_FM", np.linspace(-15, -15, 1) * 182 * kHz, inverse=True)
    zeeman_slower_FM = 0
    loading_time = 5000 * ms
    # loading_time = np.linspace(0, 19, 20)

    hold_time = 12  * ms
    shutdown_time = 1 * ms
    exposure_time = .75 * ms

    resonance_FM_556 = cal("MOT_YB_3D_556_FM", -13 * 182 * kHz, inverse=True)
    
    tof = np.linspace(1, 20, 10) * ms
    t = 0

    # --- FUNCTIONS --- #

    def load_mot(t, frequency):

        # 3D MOT cooling light
        exp.go_low(*MOT_YB_3D_399_TTL, t=t)
        exp.linramp(*MOT_YB_3D_399_AM, t=t, duration=1 * ms, start_val=0, end_val=loading_AM_399, keep_val=True)
        exp.linramp(*MOT_YB_3D_399_FM, t=t, duration=1 * ms, start_val=0, end_val=loading_FM_399, keep_val=True)
        exp.go_low(*MOT_YB_3D_556_TTL, t=t)
        exp.linramp(*MOT_YB_3D_556_AM, t=t, duration=1 * ms, start_val=0, end_val=loading_AM_556, keep_val=True)
        exp.linramp(*MOT_YB_3D_556_FM, t=t, duration=1 * ms, start_val=0, end_val=frequency, keep_val=True)

        # Zeeman slower light
        exp.go_low(*MOT_YB_ZEEMAN_399_TTL, t=t)
        exp.linramp(*MOT_YB_ZS_399_FM, t=t, duration=1 * ms, start_val=0, end_val=zeeman_slower_FM, keep_val=True)


        # 2D MOT cooling light
        exp.go_low(*MOT_YB_2D_399_TTL, t=t)
        exp.linramp(*MOT_YB_2D_399_FM, t=t, duration=1 * ms, start_val=0, end_val=0, keep_val=True)

        t += 1 * ms

        return t
    
    def fluorescence_image(t, frequency=resonance_FM_556, amplitude=loading_AM_556):

        # Fluorescence image with 556 nm light
        exp.linramp(*MOT_YB_3D_556_AM, t=t, duration=50 * us, start_val=amplitude, end_val=0, keep_val=True)
        exp.linramp(*MOT_YB_3D_556_FM, t=t, duration=50 * us, start_val=frequency, end_val=frequency, keep_val=True) 
        t += 100 * us
        exp.go_low(*MOT_YB_3D_556_TTL, t=t)
        camera.trig(time=t, exp_time=exposure_time)
        t += exposure_time
        exp.go_high(*MOT_YB_3D_556_TTL, t=t)

        return t
    
    # --- MAIN SEQUENCE --- #

    for dt in tof:
        for detuning in loading_FM_556:

            # Turn on main coils
            exp.linramp(*MOT_COIL, t=t, duration=10 * ms, start_val=0, end_val=loading_gradient, keep_val=True) 
            exp.linramp(*MOT_COIL_BIAS_ARM_Z, t=t, duration=1 * ms, start_val=0, end_val=loading_bias_Z, keep_val=True)
            exp.linramp(*MOT_COIL_BIAS_ARM_A, t=t, duration=1 * ms, start_val=0, end_val=loading_bias_A, keep_val=True)
            exp.linramp(*MOT_COIL_BIAS_ARM_C, t=t, duration=1 * ms, start_val=0, end_val=loading_bias_C, keep_val=True)

            # Turn off MOT for 100 ms
            exp.go_high(*MOT_YB_3D_399_TTL, t=t)
            exp.go_high(*MOT_YB_3D_556_TTL, t=t)
            t += 100e-3

            # Load cs MOT
            t = load_mot(t, detuning)
            t += loading_time

            # Turn off 399 nm, ramp down gradient and wait
            exp.go_high(*MOT_YB_3D_399_TTL, t=t)
            t += hold_time

            # Turn off MOT
            exp.linramp(*MOT_COIL, t=t, duration=shutdown_time, start_val=loading_gradient, end_val=0, keep_val=True)
            exp.go_high(*MOT_YB_3D_399_TTL, t=t)
            exp.go_high(*MOT_YB_3D_556_TTL, t=t)
            exp.go_high(*MOT_YB_ZEEMAN_399_TTL, t=t)
            exp.go_high(*MOT_YB_2D_399_TTL, t=t)
            t += 2.6e-3

            # Take TOF images
            t += dt
            t = fluorescence_image(t)

            # Buffer time
            t += 100 * ms

    seq_name, seq_path = compile_sequence()
    camera.start(save_to=f'{seq_path}/data', launch_live_analysis=True, seq_name=seq_name, num_reps=5)
    run_sequence(seq_name=seq_name, nreps=1) 
    camera.stop()
    print(seq_name)