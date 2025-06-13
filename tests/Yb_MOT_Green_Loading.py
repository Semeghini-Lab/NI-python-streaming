from setup_py import setup_ni
from calibration_py import setup_calibration

import numpy as np
import os, sys, time
sys.path.append(r"Z:/sequences/helper-library")
sys.path.append(r"Z:/alvium-camera")
from alvium import AlviumCamera

if __name__ == '__main__':

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
    kHz = 1e3
    MHz = 1e6

    loading_gradient = cal('MOT_COIL', 25.781, inverse=True)
    loading_bias_Z = 9.5
    loading_bias_A = 3
    loading_bias_C = 7
    loading_AM_399 = 0
    loading_AM_556 = 0
    loading_FM_399 = cal("MOT_YB_3D_399_FM", -50 * MHz, inverse=True)
    loading_FM_556 = cal("MOT_YB_3D_556_FM", -25 * 182 * kHz, inverse=True)
    loading_time = np.linspace(100, 4000, 20) * ms
    # loading_time = np.linspace(0, 19, 20)

    exposure_time = 10 * ms
    t = 0

    # --- FUNCTIONS --- #

    def load_mot(t):

        # 3D MOT cooling light
        exp.MOT_YB_3D_399_TTL.low(t=t)
        exp.MOT_YB_3D_399_AM.linramp(t=t, duration=1 * ms, start=0, end=loading_AM_399)
        exp.MOT_YB_3D_399_FM.linramp(t=t, duration=1 * ms, start=0, end=loading_FM_399)
        exp.MOT_YB_3D_556_TTL.low(t=t)
        exp.MOT_YB_3D_556_AM.linramp(t=t, duration=1 * ms, start=0, end=loading_AM_556)
        exp.MOT_YB_3D_556_FM.linramp(t=t, duration=1 * ms, start=0, end=loading_FM_556)

        # Zeeman slower light
        exp.MOT_YB_ZEEMAN_399_TTL.low(t=t)

        # 2D MOT cooling light
        exp.MOT_YB_2D_399_TTL.low(t=t)
        exp.MOT_YB_2D_399_FM.linramp(t=t, duration=1 * ms, start=0, end=0)

        t += 1 * ms

        return t
    
    # --- MAIN SEQUENCE --- #

    for dt in loading_time:

        # Turn on main coils
        exp.MOT_COIL.linramp(t=t, duration=10 * ms, start=0, end=loading_gradient) 
        exp.MOT_COIL_BIAS_ARM_Z.linramp(t=t, duration=1 * ms, start=0, end=loading_bias_Z)
        exp.MOT_COIL_BIAS_ARM_A.linramp(t=t, duration=1 * ms, start=0, end=loading_bias_A)
        exp.MOT_COIL_BIAS_ARM_C.linramp(t=t, duration=1 * ms, start=0, end=loading_bias_C)

        # Turn off MOT for 100 ms
        exp.MOT_YB_3D_399_TTL.high(t=t)
        exp.MOT_YB_3D_556_TTL.high(t=t)
        t += 100e-3

        # Load cs MOT
        t = load_mot(t)
        t += dt

        # Turn off 399 nm, ramp down gradient and wait
        exp.MOT_YB_3D_399_TTL.high(t=t)
        t += 50 * ms

        # Imaging
        camera.trig(time=t, exp_time=exposure_time)

        # Buffer time
        t += 50 * ms

    exp.compile()
    streamer = exp.create_streamer(
        num_workers=2,
        num_writers=2,
        pool_size=8
    )

    seq_path=os.path.abspath(os.path.join(os.path.dirname(__file__), 'Yb_MOT_Green_Loading'))
    seq_name='test-yb-green'
    camera.start(save_to=f'{seq_path}', launch_live_analysis=True, seq_name=seq_name, num_reps=1)

    streamer.start()
    camera.stop()
    print(seq_name)