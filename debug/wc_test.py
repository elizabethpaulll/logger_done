import cv2
import pickle
from queue import SimpleQueue, Empty
from datetime import datetime
from cv2 import INTER_AREA
from threading import Thread
from tqdm import tqdm
import os
import time

"""
Depricated version of the webcamstream class. This version is not used in the final implementation of the project but was kept test out additional capturing methods to deal with multiple cameras at the same time.
"""

class WebcamStream:
    """
    This class sets up a webcam stream using the specified source webcam and 
    reads the frames on a separate thread for better performance.
    """
    def __init__(self, id, img_save_path, log_save_path, fps, frame_size, buffer_size=None, debug=False, show=False) -> None:
        """
        source in [0..n], n = max. avalaible cameras
        """
        
        self.id = str(id)
        self.name = f"cam_{self.id}"
        self.path = img_save_path+f"{self.name}_frames/"
        if not os.path.exists(self.path): os.makedirs(self.path)
        if not os.path.exists(log_save_path): os.makedirs(log_save_path)
        self.log_path = log_save_path+f"{self.name}_log.csv"
        self.log_file = open(f"{self.log_path}", "w")
        if os.path.isfile(self.path): self.camera_log_file = open(f"{self.log_path}", "a")
        self.fps = fps
        self.stream_buffer = SimpleQueue()
        self.ping_rate = (1/(fps))
        self.buffer_size = buffer_size
        self.frame_size = frame_size
        self.stream = cv2.VideoCapture(int(self.id))
        # self.out = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.frame_size)
        #(self.read, self.frame) = self.stream.read()
        self.stopped = False
        self.show_cam = show
        self.ready_state = False
        self.debug = debug
        self.debug_base = f"[CAM {self.id}]: "
        self.permission = False

        self.log_header = f"({self.name}) read_success"
        self.log_buffer = SimpleQueue()

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
        self.setup()
        frame_id = 0
        self.log_file.write(f"timestamp;{self.name}_success;{self.name}_paths\n")
        # for i in range(3):
        #     Thread(target=self.__write__, args=()).start()
        while not self.stopped:
            start = time.time()
            (self.read, self.frame) = self.stream.read()
            self.frame_name = self.path+str(f"frame_{frame_id}.jpg")
            self.log_buffer.put((datetime.now(), str(self.read), self.frame_name))
            if not self.read or not self.stream.isOpened():
                print(f"{self.debug_base}Frame could not be read on camera. Stopping webcam stream for camera {self.id}.")
                self.stop()
                break
            else:
                self.stream_buffer.put((self.frame_name, self.frame))
            frame_id += 1
            #print(time.time()-start)

    def __write__(self):
        
        while not self.stopped:
            try:
                entry = self.stream_buffer.get()
                cv2.imwrite(entry[0], entry[1])
                log_entry = self.log_buffer.get()
                log_line = f"{log_entry[0]};{log_entry[1]};{log_entry[2]}\n"
                self.log_file.write(log_line)
            except Empty:
                break

        remaining = self.stream_buffer.qsize()
        #print(f"{remaining} in img queue.")
        #if remaining > 0: print(f"{self.debug_base}Finishing write up..")
        for i in tqdm(range(remaining), desc=f"{self.debug_base}Writing remaining items..", colour="BLUE"):
            #print("inside image finishing")
            try:
                entry = self.stream_buffer.get()
                cv2.imwrite(entry[0], entry[1])
                log_entry = self.log_buffer.get()
                log_line = f"{log_entry[0]};{log_entry[1]};{log_entry[2]}\n"
                self.log_file.write(log_line)
            except Empty:
                break
    
    def setup(self):
        cam_open = False
        frame_readable = False
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        self.stream.set(cv2.CAP_PROP_FPS, self.fps)

        while not self.permission:
            if not self.ready_state:
                cam_open = self.stream.isOpened()
                frame_readable, _ = self.stream.read()

                if cam_open and frame_readable:
                    self.ready_state = True 
             

    def stop(self):
        """
        Function that sets stopped flag true. Will stop the thread and the
        stream. Use if logging is finished
        """
        self.stopped = True
    
    def set_permission(self, permission):
        self.permission = permission




def load_ex():
    """
    Helper function, mainly for debugging. Loads videos from a file location and converts them to pickle file to
    test if it's more feasible to save them as .pkl instead of .mp4
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

if __name__ == "__main__":
    #main thread example
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    q = SimpleQueue()
    l = []
    start = time.time()
    while (time.time()-start) <= 10:
        current = time.time()
        ret, frame = cap.read()
        q.put(frame)
        l.append(frame)
        #print(time.time()-current)
        
    print(len(l))
    print(q.qsize())
    ##############################
    #thread example
    time.sleep(5)
    cam = WebcamStream(0, "./images/", "./log/", 30, (1920, 1080), 2, debug=True, show=False)
    cam.start()
    while not cam.ready_state:
        print("waiting")
        time.sleep(2)
    cam.set_permission(True)
    time.sleep(10)
    cam.stop()
    print(cam.stream_buffer.qsize())