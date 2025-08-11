import cv2
import pickle
#from multiprocessing import Process, SimpleQueue
from queue import SimpleQueue, Empty
from datetime import datetime
from cv2 import INTER_AREA
from threading import Thread
from tqdm import tqdm
import os
import time

"""
The `WebcamStream` class is a multi-threaded solution for capturing and saving webcam frames along with logging metadata. It ensures efficient frame acquisition and disk writing while providing debugging and monitoring capabilities.

### Key Features

1. **Initialization**:
   - Configures directories for image and log file storage based on camera ID.
   - Sets up webcam stream parameters like FPS, frame size, and buffer size.

2. **Streaming**:
   - Captures frames continuously on a separate thread to avoid blocking the main program.
   - Maintains a buffer to store frames temporarily before writing to disk.

3. **Logging**:
   - Logs frame metadata (timestamp, read success, file path) to a CSV file.
   - Uses a separate thread to write log data from a buffer to ensure non-blocking operations.

4. **Image Writing**:
   - Saves frames to disk in JPEG format with adjustable quality settings.
   - Writing operations are performed on a separate thread to improve efficiency.

5. **Thread Management**:
   - Handles multi-threaded operations for frame acquisition, logging, and image saving.
   - Supports dynamic adjustments to thread sleep time based on system performance.

6. **Setup and Synchronization**:
   - Waits for camera readiness and a permission flag to ensure all streams start logging synchronously.
   - Provides methods to retrieve camera FPS and resolution for debugging purposes.

7. **Debugging Tools**:
   - Logs frame processing times for performance monitoring.
   - Includes a helper function (`load_ex`) to test saving and loading frames from pickle files.

### Workflow

1. **Initialization**:
   - Creates directories for image and log files if they don’t exist.
   - Configures webcam stream properties such as frame size, FPS, and buffer size.

2. **Frame Capture**:
   - Captures frames in a dedicated thread and pushes them into a stream buffer.
   - Logs frame metadata into a log buffer.

3. **Writing Data**:
   - Separate threads handle:
     - Writing frames from the stream buffer to disk as JPEG files.
     - Writing log entries from the log buffer to a CSV file.

4. **Synchronization**:
   - Ensures cameras are ready before starting logging through a permission flag.
   - Dynamically adjusts to system performance using an adaptive drift mechanism.

5. **Shutdown**:
   - Stops threads and releases camera resources when logging is complete.

### Debugging and Testing

The class includes utilities for monitoring performance and testing alternative data storage methods. For example:
- The `load_ex` function helps test frame storage as pickle files for debugging.
- Debug logs track frame processing times and system performance.
"""


class WebcamStream:
    """
    This class sets up a webcam stream using the specified source webcam and 
    reads the frames on a separate thread for better performance.
    """
    def __init__(self, id, img_save_path, log_save_path, fps, frame_size, buffer_size=None, debug=False, show=False, writer_threads=1, quality=75) -> None:
        """
        source in [0..n], n = max. avalaible cameras
        """
       
        #file init
        self.id = id
        self.name = f"cam_{self.id}"
        self.path = img_save_path+f"{self.name}_frames/"
        if not os.path.exists(self.path): os.makedirs(self.path)
        self.log_path = log_save_path+f"{self.name}_log.csv"
        if os.path.isfile(self.path): self.camera_log_file = open(f"{self.log_path}", "a")
        else: self.log_file = open(f"{self.log_path}", "w") 

        #stream init
        self.fps = fps
        self.stream_buffer = SimpleQueue()
        self.ping_rate = (1/(fps))
        self.buffer_size = buffer_size
        self.frame_size = frame_size
        self.stream = cv2.VideoCapture(int(self.id))
        self.stopped = False
        self.show_cam = show
        self.ready_state = False
        #debug and cleanup init
        self.debug = debug
        self.debug_base = f"[CAM {self.id}]: "
        self.permission = False
        self.check_sum = 0
        self.writer_threads = writer_threads
        self.quality = quality

        self.log_header = f"({self.name}) read_success"
        self.log_buffer = SimpleQueue()
        self.debug_frequency_log = []
        self.quality = quality
        self.write_limit = 60
        self.writer_sleep_time = 3
        self.adaptive_drift = 0

    def start(self):
        """
        Starts the thread and runs get method on it.
        """
        Thread(target=self.__get__, args=()).start()
        return self

    def __get__(self):
        """
        Function that reads frames from the webcam stream. This is run
        on thread. Other code that is meant to be run in that thread goes
        here.
        """
        
        #init cams
        self.setup()
        #init frame id
        frame_id = 0
        #init csv file (set column headers)
        self.log_file.write(f"timestamp;{self.name}_success;{self.name}_paths\n")

        #init writing threads
        # for i in range(self.writer_threads):
        Thread(target=self.__write_log__, args=()).start()
        Thread(target=self.__write_img__, args=()).start()

        #set timing
        #last_time = time.time()
        #data loop
        while not self.stopped:

                current_time = time.time()
            # if current_time-last_time>=self.ping_rate: #if ping rate reached --> get data
                
                (self.read, self.frame) = self.stream.read()
                self.frame_name = self.path+str(f"frame_{frame_id}.jpg")

                #put current log line into buffer
                self.log_buffer.put(";".join([str(datetime.now()), str(self.read), self.frame_name])+"\r")

                #error message if streams fail
                if not self.read or not self.stream.isOpened():
                    print(f"{self.debug_base}Frame could not be read on camera. Stopping webcam stream for camera {self.id}.")
                    self.stop()
                    break
                else:
                    #put images and related path in streaming buffer for writer threads to save them
                    self.stream_buffer.put((self.frame_name, self.frame))
                    frame_id += 1
                    #self.check_writing()
                self.debug_frequency_log.append((str(datetime.now()),time.time()-current_time))
                
                
        #self.check_writing()
                   
    def check_writing(self):
        if self.stream_buffer.qsize()>=self.write_limit:
            Thread(target=self.__write_img__(), args=()).start()
        if self.log_buffer.qsize()>=self.write_limit:
            Thread(target=self.__write_log__(), args=()).start()

    def __write_log__(self):

        while not self.stopped:
            try:
                    #last = time.time()
                #if self.log_buffer.qsize()>=self.write_limit:
                    limit = self.log_buffer.qsize()
                    for i in range(limit):
                        log_line = self.log_buffer.get()
                        self.log_file.write(log_line)
            except Empty:
                continue
            time.sleep(self.writer_sleep_time)
        
        while self.log_buffer.qsize()>0:
            try:
                log_line = self.log_buffer.get()
                self.log_file.write(log_line)
            except Empty:
                break



    def __write_img__(self):
        while not self.stopped:
            try:
                    last = time.time()
                #if self.stream_buffer.qsize()>=self.write_limit: 
                    limit = self.stream_buffer.qsize() # + int(self.adaptive_drift)
                    for i in range(limit): #write data remaining in stream buffer
                        entry = self.stream_buffer.get()
                        cv2.imwrite(entry[0], entry[1], [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            except Exception as e:
                continue
            #self.adaptive_drift = (time.time() - last) * 30.0 #dynamically adjust thread sleep time according to I/O speed 
            time.sleep(self.writer_sleep_time)
        
        while self.stream_buffer.qsize()>0:
            try:
                entry = self.stream_buffer.get()
                cv2.imwrite(entry[0], entry[1], [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            except Exception as e:
                print(f"{self.debug_base} write error\n", e)
                break

        

    
    def setup(self):
        cam_open = False
        frame_readable = False
        #camera params
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        self.stream.set(cv2.CAP_PROP_FPS, self.fps)

        #wait until frame can be retrieved and controller changes permission flag to start logging at the same time
        while not self.permission:
            if not self.ready_state:
                cam_open = self.stream.isOpened()
                frame_readable, _ = self.stream.read()

                if cam_open and frame_readable:
                    self.ready_state = True 
             
    def get_fps(self):
        return self.stream.get(cv2.CAP_PROP_FPS)

    def get_size(self):
        return f"{int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    
    def resized_frame(self, frame):
        return cv2.resize(frame, (self.frame_size[0], self.frame_size[1]), interpolation=INTER_AREA)

    def stop(self):
        """
        Function that sets stopped flag true. Will stop the thread and the
        stream.
        """
        self.stopped = True
    
    def set_permission(self, permission):
        self.permission = permission


####################### Debugging & Test functions ##############################

def load_ex():
    """
    Helper function, mainly for debugging. Loads videos from a file location and converts them to pickle file to
    test if it's more feasible to save them as .pkl instead of .mp4 --> nope
    """
    save_folder = "./images/"
    file_f = open(save_folder+"frontal/frontal.pkl", 'rb')
    file_s = open(save_folder+"side_profile/side.pkl", 'rb')
    buffer_frontal = pickle.load(file_f)
    buffer_side = pickle.load(file_s)

    id = 0
    for i in range(len(buffer_frontal)):
        cv2.imwrite(save_folder+"frontal/frame_f_%d.jpg" % id, buffer_frontal[i])
        cv2.imwrite(save_folder+"side_profile/frame_s_%d.jpg" % id, buffer_side[i])
        id += 1
