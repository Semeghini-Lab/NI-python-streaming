import os
import numpy as np
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py
import shutil
import time
import multiprocessing
from vmbpy import *
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import json

'''
AlviumCamera is the class that the user interacts with to control the camera.
Behind the scenes, AlviumCamera also tracks and computes the metadata to be saved with each image.
AlviumCamera starts the CameraStreamWorker and LiveAnalysisWorker processes.
'''
class AlviumCamera:

    def __init__(self, exp, cam_id, CAM_TRIG, metadata, gain=0, h5_file=r'C:/imgs'):
        '''
        Parameters:
        exp: Experiment object
        cam_id: name of the camera; e.g. 'DEV_1AB22C060BE3'
        CAM_TRIG: channel of the camera trigger
        metadata (dict): list of variables; each variable will be saved as metadata with the image files.
            e.g.:
            metadata = {'MOT_COIL': MOT_COIL,
                        'MOT_3D_COOLING_FM': MOT_3D_COOLING_FM}
        h5_file (str): path to the folder where the h5 file should be saved
        '''
        self.exp = exp
        self.cam_id = cam_id
        self.CAM_TRIG = CAM_TRIG
        self.metadata = metadata
        self.gain = gain
        self.h5_file = h5_file
        
        self.meta_names = []            # name of each piece of metadata
        self.meta_channels = []         # corresponding NI channel to query its value
        self.image_start_times = []     # list of start times each image is taken
        self.image_end_times = []       # list of end times each image is taken (i.e. start time + exposure time)
        self.meta_values = []           # list of metadata values for each image: [[val1, val2, ...], [val1, val2, ...], ...]
        for key in metadata.keys():
            self.meta_names.append(key)
            self.meta_channels.append(metadata[key])

        # by default, also keep track of the time, exposure time, and repetition index of each image
        self.meta_names.extend(['t', 'exposure_time', 'rep_index'])
    
    def trig(self, time, exp_time):
        '''
        Add into the experiment control sequence a trigger to take an image with the Alvium camera.
        The length of the trigger pulse is the exposure time of the camera.

        Parameters:
        time (float): time at which the image was taken
        exp_time (float): exposure time of the camera
        '''
        self.CAM_TRIG.high(t=time)
        self.CAM_TRIG.low(t=time+exp_time)
        self.image_start_times.append(time)
        self.image_end_times.append(time+exp_time)

    def start(self, save_to, seq_name, num_reps=1, launch_live_analysis=True):
        '''
        Call this function after the experiment has been compiled and after all calls to trig() have been made.
        Parameters:
        save_to (str): Destination folder
        seq_name (str): A sequence name that is used as the plot title.
        num_reps (int): Number of repetitions of the image sequence.
        launch_live_analysis (bool): if True, start a live-analysis GUI.
        '''
        # Store the destination folder for later use.
        self.save_to = save_to
        self.h5_file = os.path.join(self.h5_file, f"{seq_name}-images.h5")
        self.live_analysis_process = None
        
        # Calculate metadata values for each image over all repetitions.
        if not self.exp.is_compiled:
            raise RuntimeError("Experiment is not compiled. Call exp.compile() before calling start().")
        
        self.meta_values.clear()
        for rep in range(num_reps):
            for i in range(len(self.image_start_times)):
                vals = []
                st = self.image_start_times[i]
                et = self.image_end_times[i]

                # Also include time (midpoint), exposure time, and repetition index.
                t = (st + et) / 2
                exposure_time = et - st
                for channel in self.meta_channels:
                    # Use channel_calc_signal_nsamps to compute a representative value during exposure.
                    vals.append(channel(t))

                vals.extend([t, exposure_time, rep])
                self.meta_values.append(vals)
        
        # Determine the total number of images to be acquired.
        total_images = len(self.image_start_times) * num_reps

        # Spawn the worker process to handle the camera stream.
        self.stream_stop_event = multiprocessing.Event()
        stream_ready_event = multiprocessing.Event()
        self.camera_stream_process = CameraStreamWorker(self.cam_id, self.h5_file, self.meta_names, self.meta_values,
                                                        total_images, self.gain, self.stream_stop_event, stream_ready_event)
        self.camera_stream_process.start()
        # Wait until the camera stream has been set up.
        stream_ready_event.wait()

        # Launch the live analysis process (reads from the HDF5 file via SWMR).
        if launch_live_analysis:
            self.live_analysis_process = LiveAnalysisWorker(self.h5_file, self.stream_stop_event, seq_name)
            self.live_analysis_process.start()

    def stop(self):
        self.stream_stop_event.set()
        self.camera_stream_process.join()
        if self.live_analysis_process:
            # if live analysis is running, give live analysis 1 second to close the HDF5 file before moving
            time.sleep(1)
        dest_path = os.path.join(self.save_to, os.path.basename(self.h5_file))
        shutil.move(self.h5_file, dest_path)
        del self


'''
Worker process to handle the camera stream.
This class opens a single HDF5 file (using libver='latest') and writes each acquired image and its metadata
to two preallocated datasets. A new dataset ("frame_count") is used to track how many images have been acquired.
Once the datasets are created the file is switched to SWMR mode.
'''
class CameraStreamWorker(multiprocessing.Process):
    def __init__(self, cam_id, h5_file, meta_names, meta_values, total_frames, gain, stream_stop_event, stream_ready_event):
        super().__init__()
        self.cam_id = cam_id
        self.h5_file = h5_file
        self.meta_names = meta_names
        self.meta_values = meta_values
        self.meta_n = len(meta_names)
        self.total_frames = total_frames
        self.gain = gain
        self.stream_stop_event = stream_stop_event
        self.stream_ready_event = stream_ready_event
        self.frame_count = 0
        self.f = None
        self.dataset_images = None
        self.dataset_metadata = None
        self.dataset_frame_count = None

    def run(self):
        # Open the HDF5 file for writing.
        self.f = h5py.File(self.h5_file, "w", libver="latest")

        self.f.attrs['meta_names'] = json.dumps(self.meta_names)
        self.f.attrs['meta_values'] = json.dumps(self.meta_values)
        
        with VmbSystem.get_instance() as vmb:
            cams = vmb.get_all_cameras()
            n = len(cams)
            if n==0:
                raise RuntimeError('No Allied Vision cameras found.')
            
            # find the desired camera
            index = -1
            for i in range(n):
                if cams[i].get_id() == self.cam_id:
                    index = i
                    break
            if index==-1:
                raise RuntimeError(f'Could not find Allied Vision camera with id {self.cam_id}')

            with cams[index] as cam:
                # Setup the camera for external triggering.
                cam.ExposureMode.set('TriggerWidth')
                cam.LineSelector.set('Line0')
                cam.LineMode.set('Input')
                cam.TriggerSelector.set('FrameStart')
                cam.TriggerActivation.set('LevelHigh')
                cam.TriggerSource.set('Line0')  # Use 'Line0' for external trigger
                cam.TriggerMode.set('On')  # Enable external triggering
                # cam.AcquisitionMode.set('Continous')  # Capture one frame per trigger
                cam.Gain.set(self.gain)

                print('Waiting for external trigger to capture a frame.')
                self.stream_ready_event.set()
                try:
                    cam.start_streaming(handler=self.frame_handler,
                                          buffer_count=10,
                                          allocation_mode=AllocationMode.AllocAndAnnounceFrame)
                    while not self.stream_stop_event.is_set():
                        time.sleep(0.2)
                finally:
                    cam.stop_streaming()
                    self.f.flush()
                    self.f.close()

    def frame_handler(self, cam, stream, frame):
        # When a frame is acquired, convert it to a numpy array.
        print(f'Frame {self.frame_count} acquired', end='\r')
        assert frame.get_status() == FrameStatus.Complete, 'Incomplete frame captured from Alvium camera. Frame rate may be too high.'

        image_data = frame.as_numpy_ndarray()

        # On the first frame, create preallocated datasets for images, metadata, and frame_count.
        if self.dataset_images is None:
            image_shape = image_data.shape
            total = self.total_frames
            self.dataset_images = self.f.create_dataset(
                "images",
                shape=(total,) + image_shape,
                maxshape=(total,) + image_shape,
                chunks=(1,) + image_shape,
                dtype=image_data.dtype
            )
            self.dataset_metadata = self.f.create_dataset(
                "metadata",
                shape=(total, self.meta_n),
                maxshape=(total, self.meta_n),
                chunks=(1, self.meta_n),
                dtype='float64'
            )
            # Create a dataset to hold the current frame count.
            self.dataset_frame_count = self.f.create_dataset(
                "frame_count",
                shape=(1,),
                maxshape=(1,),
                dtype='int64'
            )
            self.dataset_frame_count[0] = 0
            # Now that the datasets are created, enable SWMR mode.
            self.f.swmr_mode = True

        # Write the image and metadata at the corresponding index.
        self.dataset_images[self.frame_count, ...] = image_data
        metadata_row = self.meta_values[self.frame_count]
        self.dataset_metadata[self.frame_count, :] = metadata_row

        self.frame_count += 1
        # Update the frame_count dataset so that live analysis can know how many frames are valid.
        self.dataset_frame_count[0] = self.frame_count
        self.f.flush()  # Ensure data is written so SWMR readers see the update.
        cam.queue_frame(frame)


'''
Worker process for live analysis.
This process opens the same HDF5 file in read-only SWMR mode and periodically refreshes its view.
It now reads the "frame_count" dataset (after refreshing the "images" and "frame_count" datasets) 
to determine how many images have been acquired so far.
'''
class LiveAnalysisWorker(multiprocessing.Process):
    def __init__(self, h5_file, stream_stop_event, seq_name):
        super().__init__()
        self.h5_file = h5_file
        self.stream_stop_event = stream_stop_event
        self.seq_name = seq_name
        self.last_image = None        
        self.indices = []
        self.counts = []
        # Use last_index to track the last processed image index.
        self.last_index = 0

    def run(self):
        # Wait until the file exists and the "images" dataset is available.
        f = None
        while f is None:
            try:
                f = h5py.File(self.h5_file, 'r', swmr=True)
                if "images" not in f:
                    f.close()
                    f = None
            except Exception:
                f = None
            time.sleep(0.1)

        # Now that the datasets exist, proceed with the live viewer.
        root = tk.Tk()
        root.title("Live Image Analysis")
        fig = Figure(figsize=(15, 10), dpi=100)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        ds_images = f["images"]
        ds_frame_count = f["frame_count"]

        def update_gui():
            ds_frame_count.refresh()
            num_valid_images = int(ds_frame_count[0])
            
            # Process new images starting from the last processed index.
            if num_valid_images > self.last_index:
                ds_images.refresh()
                for i in range(self.last_index, num_valid_images):
                    img = ds_images[i]
                    self.indices.append(i)
                    self.counts.append(np.sum(img))
                self.last_index = num_valid_images
                self.last_image = ds_images[num_valid_images-1]

            if self.counts:
                ax1.clear()
                ax1.plot(self.indices, self.counts, '.-k')
                ax1.set_title(self.seq_name)
                ax1.set_xlabel("Image Index")
                ax1.set_ylabel("Raw Counts")

            if self.last_image is not None:
                ax2.clear()
                ax2.imshow(self.last_image, cmap='gray')
                ax2.set_title("Most Recent Image")
                ax2.axis('off')

            canvas.draw()
            if self.stream_stop_event.is_set():
                f.close()
            else:
                root.after(200, update_gui)

        def on_closing():
            if self.stream_stop_event.is_set():
                root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.after(200, update_gui)
        root.mainloop()
