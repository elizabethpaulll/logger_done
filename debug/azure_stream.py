import cv2
from utils import joint_names, get_joint_coordinates, format_coordinates
from queue import SimpleQueue, Empty
from datetime import datetime
from cv2 import INTER_AREA
from threading import Thread
import time
import os
import pykinect_azure as pykinect

'''
For body pose, infrared, and depth frame logging, a custom azure kinect stream was implemented building on the pykinect_azure library. It can be called as follows:

    stream = AzureKinectStream(log_save_path, img_path, fps, debug=False, writer_threads=3)
    stream.start() 
    stream.set_permission(True) <-- this starts the actual streaming
    stream.stop()

All cameras used in the logging script are meant to run a their own threads for faster processing. Any changes in the way the azure kinect collects images should be done in this file.
'''

class AzureKinectStream:
    """
    This class sets up a azure kinect stream using the specified source and 
    reads color, depth, ir, and keypoints.
    """
    def __init__(self, log_save_path, img_path, fps, debug=False, writer_threads=3) -> None:
        """
        Separate kinect stream to avoid kinect being recognized as regular webcam
        """
        self.name = f"multi_sensor_stream"
        self.c_path = img_path+f"{self.name}_c_frames/"
        self.d_path = img_path+f"{self.name}_d_frames/"
        self.ir_path = img_path+f"{self.name}_ir_frames/"
        self.log_dir = log_save_path
        self.log_path = log_save_path+f"{self.name}_log.csv"
        self.__path_config__()
        self.log_file = open(f"{self.log_path}", "w")
        if os.path.isfile(self.log_path): self.camera_log_file = open(f"{self.log_path}", "a")
        self.stream_buffer = SimpleQueue()
        self.fps = fps
        self.ping_rate = (1/(fps))
        self.stopped = False
        self.ready_state = False
        self.debug = debug
        self.debug_base = f"[AZURE KINECT]: "
        self.permission = False
        self.check_sum = 0
        self.ready_state = False
        self.log_header = ";".join(joint_names)+"\n"
        self.log_buffer = SimpleQueue()
        self.writer_threads = writer_threads

    def start(self):
        """
        Starts the thread and runs get method on it.
        """
        Thread(target=self.__get__, args=()).start()
        #return self
    
    def __path_config__(self):
        """
        Configure necessary paths according to participant ID
        """
        if not os.path.exists(self.c_path): os.makedirs(self.c_path)
        if not os.path.exists(self.d_path): os.makedirs(self.d_path)
        if not os.path.exists(self.ir_path): os.makedirs(self.ir_path)
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)

    def __get__(self):
        """
        Function that reads frames from the webcam stream. This is run
        on thread. Other code that is meant to be run in that thread goes
        here.
        """
        last_capture_time = time.time()
        frame_id = 0

        pykinect.initialize_libraries(track_body=True)
        # Modify camera configuration
        device_config = pykinect.default_configuration
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED


        device = pykinect.start_device(config=device_config)
        body_tracker = pykinect.start_body_tracker()
        self.log_file.write(f"timestamp;color_success;depth_success;ir_success;{self.name}_c_paths;{self.name}_d_paths;{self.name}_ir_paths;{self.log_header}\r")


        # Start writing thread
        while not self.stopped:
            #current_time = time.time()
            capture = device.update()
            body_frame = body_tracker.update()
            if not self.permission:
                self.ret_color, self.c_image = capture.get_color_image()			
                self.ret_ir, self.ir_image = capture.get_ir_image()
                self.ret_depth, self.d_image = capture.get_depth_image()
                self.joint_coords = format_coordinates(get_joint_coordinates(body_frame))

                if self.ret_color and self.ret_ir and self.ret_depth:
                    self.ready_state = True 
            else:        
                #if current_time-last_capture_time>=self.ping_rate:
                    self.ret_color, self.c_image = capture.get_color_image()			
                    self.ret_ir, self.ir_image = capture.get_ir_image()
                    self.ret_depth, self.d_image = capture.get_depth_image()
                    self.joint_coords = format_coordinates(get_joint_coordinates(body_frame))

                    self.c_name = self.c_path+str(f"azure_c_frame_{frame_id}.jpg")
                    self.d_name = self.d_path+str(f"azure_d_frame_{frame_id}.jpg")
                    self.ir_name = self.ir_path+str(f"azure_ir_frame_{frame_id}.jpg")
                    
                    self.log_buffer.put(";".join([str(datetime.now()), 
                                        str(self.ret_color), 
                                        str(self.ret_ir), 
                                        str(self.ret_depth), 
                                        str(self.c_name), 
                                        str(self.d_name), 
                                        str(self.ir_name)])+self.joint_coords+"\r")
                    
                    self.stream_buffer.put([self.c_name, self.c_image, self.d_name, self.d_image, self.ir_name, self.ir_image])
                    frame_id += 1

                #last_capture_time = current_time

    def __write_log__(self):
        while not self.stopped:
            try:
                log_line = self.log_buffer.get()
                self.log_file.write(log_line)
            except Empty:
                continue

        remaining = self.log_buffer.qsize()
        for i in range(remaining):
            try:
                log_line = self.log_buffer.get()
                self.log_file.write(log_line)
            except Empty:
                break
        
    def __write_img__(self):
        while not self.stopped:
            try:
                entry = self.stream_buffer.get()
                c_path = entry[0]
                c_img = entry[1]
                d_path = entry[2]
                d_img = entry[3]
                ir_path = entry[4]
                ir_img = entry[5]
                cv2.imwrite(c_path, c_img)
                cv2.imwrite(d_path, d_img)
                cv2.imwrite(ir_path, ir_img)
            except Empty:
                continue

        remaining = self.stream_buffer.qsize()
        for i in range(remaining):
            try:
                entry = self.stream_buffer.get()
                c_path = entry[0]
                c_img = entry[1]
                d_path = entry[2]
                d_img = entry[3]
                ir_path = entry[4]
                ir_img = entry[5]
                cv2.imwrite(c_path, c_img)
                cv2.imwrite(d_path, d_img)
                cv2.imwrite(ir_path, ir_img)
            except Empty:
                break
        
    def stop(self):
        """
        Function that sets stopped flag true. Will stop the thread and the
        stream.
        """
        self.stopped = True
    
    def set_permission(self, permission):
        self.permission = permission
        #init buffering thread
        #Thread(target=self.__get__, args=()).start()
        #init writing threads
        for i in range(self.writer_threads):
            Thread(target=self.__write_log__, args=()).start()
            Thread(target=self.__write_img__, args=()).start()
  