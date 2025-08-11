import cv2
from utils import joint_names, get_joint_coordinates, format_coordinates
from queue import SimpleQueue, Empty
from datetime import datetime
from cv2 import INTER_AREA
from threading import Thread
from tqdm import tqdm
import os
import pykinect_azure as pykinect

pykinect.initialize_libraries(track_body=True)

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

class AzureKinectStream:
    """
    This class sets up a azure kinect stream using the specified source and 
    reads color, depth, ir, and keypoints.
    """
    def __init__(self, id, log_save_path, c_save_path, d_save_path, ir_save_path, fps, frame_size, buffer_size=None, debug=False, show=False) -> None:
        """
        Separate kinect stream to avoid kinect being recognized as regular webcam
        """
        
        self.id = str(id)
        self.name = f"multi_sensor_stream"
        self.c_path = c_save_path+f"{self.name}_c_frames/"
        self.d_path = d_save_path+f"{self.name}_d_frames/"
        self.ir_path = ir_save_path+f"{self.name}_ir_frames/"
        self.log_path = log_save_path+f"{self.name}_log.csv"
        self.__path_config__()
        self.log_file = open(f"{self.log_path}", "w")
        if os.path.isfile(self.path): self.camera_log_file = open(f"{self.log_path}", "a")
        self.device = pykinect.start_device(config=device_config)
        self.body_tracker = pykinect.start_body_tracker()
        self.stream_buffer = SimpleQueue()
        self.fps = fps
        self.ping_rate = (1/(fps))
        self.stopped = False
        self.ready_state = False
        self.debug = debug
        self.debug_base = f"[AZURE {self.id}]: "
        self.permission = False
        self.done_writing = 0

        self.log_header = ";".join(joint_names)+"\n"
        self.log_buffer = SimpleQueue()

    def start(self):
        """
        Starts the thread and runs get method on it.
        """
        Thread(target=self.__get__, args=()).start()
        return self
    def __path_config__(self):
        if not os.path.exists(self.c_path): os.makedirs(self.c_path)
        if not os.path.exists(self.d_path): os.makedirs(self.d_path)
        if not os.path.exists(self.ir_path): os.makedirs(self.ir_path)

    def __get__(self):
        """
        Function that reads frames from the webcam stream. This is run
        on thread. Other code that is meant to be run in that thread goes
        here.
        """
        frame_id = 0
        self.log_file.write(f"timestamp;
                            color_success;
                            depth_success;
                            ir_success;
                            {self.name}_c_paths;
                            {self.name}_d_paths;
                            {self.name}_ir_paths;
                            {self.log_header}\n")

        # Start writing thread
        for i in range(3):
            Thread(target=self.__write__, args=()).start()
        while not self.stopped:
            
            #setup condition
            if not self.permission:
                cam_open = False
                frame_readable = False

                if not self.ready_state:
                    cam_open = self.device.recording
                    capture = self.device.update()
                    frame_readable, _ = capture.get_color_image()

                if cam_open and frame_readable:
                    self.ready_state = True 

                frame_id = 0
            else:
                try:
                    self.capture = self.device.update()
                    body_frame = self.body_tracker.update()
                    ret_color, self.c_image = self.capture.get_color_image()			
                    ret_ir, self.ir_image = self.capture.get_ir_image()
                    ret_depth, self.d_image = self.capture.get_smooth_colored_depth_image()
        
                    self.c_name = self.c_path+str(f"azure_c_frame_{frame_id}.jpg")
                    self.d_name = self.d_path+str(f"azure_d_frame_{frame_id}.jpg")
                    self.ir_name = self.ir_path+str(f"azure_ir_frame_{frame_id}.jpg")
                    self.joint_coords = format_coordinates(get_joint_coordinates(body_frame))
                    
                    self.log_buffer.put(";".join([datetime.now(), 
                                        str(ret_color), 
                                        str(ret_ir), 
                                        str(ret_depth), 
                                        str(self.c_name), 
                                        str(self.d_name), 
                                        str(self.ir_name)])+self.joint_coords+"\n")
                    
                    self.stream_buffer.put([self.c_name, self.c_image, self.d_name, self.d_image, self.ir_name, self.ir_image])
                    frame_id += 1
                except:
                    continue

    def __write__(self):
        
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
                log_line = self.log_buffer.get()
                self.log_file.write(log_line)
            except Empty:
                break

        remaining = self.stream_buffer.qsize()
        print(f"{self.debug_base}: Writing remaining azure kinect data.")
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
                log_line = self.log_buffer.get()
                self.log_file.write(log_line)
            except Empty:
                break
        
        self.done_writing = 1

    def stop(self):
        """
        Function that sets stopped flag true. Will stop the thread and the
        stream.
        """
        self.stopped = True
    
    def set_permission(self, permission):
        self.permission = permission
    
    def component_cleared(self):
        return self.done_writing