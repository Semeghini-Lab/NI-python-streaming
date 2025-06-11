from nistreamer.Experiment import Experiment

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
