from setup_py import setup_ni
from calibration_py import setup_calibration

import numpy as np
import os, sys, time
sys.path.append(r"Z:/sequences/helper-library")
#sys.path.append(r"Z:/alvium-camera")
from alvium import AlviumCamera


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