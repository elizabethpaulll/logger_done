
from webcam_stream import WebcamStream
import time
import sys
import os
from tqdm import tqdm
import cv2
import pykinect_azure as pykinect
from queue import SimpleQueue, Empty #use SimpleQueue instead of queue as it is faster
from utils import joint_names, get_joint_coordinates, format_coordinates
from datetime import datetime
from threading import Thread
import statistics


"""
The `CamController` class is the central component for managing and synchronizing multiple camera streams, including Azure Kinect and webcams, in a logging script. It facilitates frame 
acquisition, logging, and image storage, ensuring high-performance and synchronized operation across devices. Key functionalities include:

1. Initialization:
   - Configures directories for saving images and log files with participant-specific paths.
   - Supports debug mode and adjustable image quality for performance optimization.

2. Camera Management:
   - Registers multiple cameras for synchronized operation.
   - Ensures all devices are ready before starting logging.

3. Data Logging:
   - Collects and logs frame data (color, depth, IR) and joint coordinates from Azure Kinect.
   - Writes image and log data to disk efficiently using separate threads for parallel processing.

4. Performance Monitoring:
   - Logs frame processing times and calculates statistics like mean and standard deviation for debugging.

5. Graceful Shutdown:
   - Stops all cameras and ensures remaining data in buffers is written to disk before exiting.

This class is designed for scenarios requiring high-fidelity, multi-camera data acquisition and logging, such as research experiments or motion analysis studies.
"""


# Video and log file paths to parent folders. Participant specific paths appended before logging.
video_base_path = 'dataset/images/'
log_base_path = 'dataset/logs/'

# number of cameras
number_of_cameras = 5
visible_cam = -1

frame_size = (1920, 1080)

fps = 30
buffer_size = 2

        
controller = ""
participant_id = ""





class CamController:
    """
    img_path = path to the folder where the images should be saved (participant specific path will be appended)
    log_save_path = path to the folder where the log files should be saved (participant specific path will be appended)
    debug = if true, debug messages will be printed
    writer_threads = number of threads that are supposed to write to the disk
    quality = jpeg quality of the images (if performance issues occur, play around with the quality and see if it helps)
    """
    def __init__(self, img_path, log_save_path, debug=True, writer_threads=2, quality=75) -> None:
        self.cams = {}
        self.setup_done = False
        self.debug_base = "[CAM CONTROLLER]: "
        self.path = log_base_path+ "log_cam_all__participant " + participant_id + ".csv"
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
        self.stopped = False
        self.ready_state = False
        self.debug = debug
        self.debug_base = f"[CONTROLLER/AZURE]: "
        self.permission = False
        self.ready_state = False
        self.log_header = ";".join(joint_names)+"\n"
        self.log_buffer = SimpleQueue()
        self.writer_threads = writer_threads
        self.kinect_setup_done = False
        self.quality = quality
        self.write_limit = 180
        self.writer_sleep_time = 10.0
        self.debug_frequency_log = []
        self.adaptive_drift = 0
        

        self.labeling = False
        # self.camera_log_file = open(log_base_path+ "log_cam_all__participant " + participant_id + ".csv", "w")
        # if os.path.isfile(self.path): self.camera_log_file = open(log_base_path+ "log_cam_all__participant " + participant_id + ".csv", "a")
    
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
        Function that reads frames from the kinect stream. This should not be run in a separate thread, must be the main thread.
        """
        frame_id = 0

        #initialize kinect
        pykinect.initialize_libraries(track_body=True)
        # Modify camera configuration
        device_config = pykinect.default_configuration
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

        # Start device and body tracker
        device = pykinect.start_device(config=device_config)
        body_tracker = pykinect.start_body_tracker()
        #write log header
        self.log_file.write(f"timestamp;color_success;depth_success;ir_success;{self.name}_c_paths;{self.name}_d_paths;{self.name}_ir_paths;{self.log_header}\r")


        # Start reading frames
        while not self.stopped:

            current_time = time.time()
            try:
                capture = device.update()
                body_frame = body_tracker.update()
            except Exception:
                continue

            #check if kinect is ready, if not, wait
            #the permission flag is set once it is ensured that frames can be read from the kinect
            if not self.permission:
                self.ret_color, self.c_image = capture.get_color_image()			
                self.ret_ir, self.ir_image = capture.get_ir_image()
                self.ret_depth, self.d_image = capture.get_depth_image()
                self.joint_coords = format_coordinates(get_joint_coordinates(body_frame))

                if self.ret_color and self.ret_ir and self.ret_depth:
                    self.ready_state = True 
            else:        
                #if frames can be read, start logging
                #if current_time-last_capture_time>=self.ping_rate:
                    self.ret_color, self.c_image = capture.get_color_image()#get color			
                    self.ret_ir, self.ir_image = capture.get_ir_image()#get infrared
                    self.ret_depth, self.d_image = capture.get_depth_image()#get depth
                    self.joint_coords = format_coordinates(get_joint_coordinates(body_frame))#get joint coordinates

                    #create paths for images
                    self.c_name = self.c_path+str(f"azure_c_frame_{frame_id}.jpg")
                    self.d_name = self.d_path+str(f"azure_d_frame_{frame_id}.jpg")
                    self.ir_name = self.ir_path+str(f"azure_ir_frame_{frame_id}.jpg")
                    #build the log line string using the read boolean values, paths, and joint coordinates
                    try:
                        self.log_buffer.put(";".join([str(datetime.now()), 
                                            str(self.ret_color), 
                                            str(self.ret_ir), 
                                            str(self.ret_depth), 
                                            str(self.c_name), 
                                            str(self.d_name), 
                                            str(self.ir_name)])+self.joint_coords+"\r")
                    except TypeError:
                        print("Skeleton out of frame. Move into the kinect range.")
                        continue
                    #put the images and paths into the stream buffer so that the writer can write them to disk
                    self.stream_buffer.put([self.c_name, self.c_image, self.d_name, self.d_image, self.ir_name, self.ir_image])
                    frame_id += 1
                    #call the writing functions to check if the buffer is sufficiently filled to write data to the disk
                    #constant writing is discouraged to minimize load on the cpu (irrelevant if multi-processing is used)
                    #if inconsistent logging rates occur in the log files, put the writing function into a separate thread and check every N seconds if there's something to write
                    #otherwise if the logging has stopped the writing functions are supposed to write consistently until the buffers have been emptied
            self.debug_frequency_log.append((str(datetime.now),time.time()-current_time))


    def check_writing(self):
        """
        Function that checks if the buffer is sufficiently filled to write data to the disk
        Stream writing and log writing have been separated as to not interfere with each other as writing images takes longer than writing csv lines
        """
        if self.stream_buffer.qsize()>=self.write_limit:
            Thread(target=self.__write_img__(), args=()).start()
        if self.log_buffer.qsize()>=self.write_limit:
            Thread(target=self.__write_log__(), args=()).start()

    def __write_log__(self):
        """
        Function that writes the log lines to the disk
        Waits for a certain number of entries: self.write_limit
        """
        while not self.stopped:
            try:
                    #last = time.time()
                #if self.log_buffer.qsize()>=self.write_limit:
                    limit = self.log_buffer.qsize()
                    for i in range(limit):
                        log_line = self.log_buffer.get()
                        self.log_file.write(log_line)
            except Exception as e:
                continue
            #drift = time.time() - last #dynamically adjust thread sleep time according to I/O speed 
            time.sleep(self.writer_sleep_time)
        
        while self.log_buffer.qsize()>0:
            try:
                log_line = self.log_buffer.get()
                self.log_file.write(log_line)
            except Exception as e:
                break
        
    def __write_img__(self):
        """
        Function that writes the log lines to the disk
        Waits for a certain number of entries: self.write_limit
        """
        while not self.stopped:
            try:
                    last = time.time()
                #if self.stream_buffer.qsize()>=self.write_limit:
                    limit = self.stream_buffer.qsize()
                    for i in range(limit):
                        entry = self.stream_buffer.get()
                        c_path = entry[0]
                        c_img = entry[1]
                        d_path = entry[2]
                        d_img = entry[3]
                        ir_path = entry[4]
                        ir_img = entry[5]
                        cv2.imwrite(c_path, c_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                        cv2.imwrite(d_path, d_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                        cv2.imwrite(ir_path, ir_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            except Exception as e:
                continue
            #drift = time.time() - last #dynamically adjust thread sleep time according to I/O speed 
            #self.adaptive_drift = (time.time() - last) * 30.0 #dynamically adjust thread sleep time according to I/O speed 
            time.sleep(self.writer_sleep_time)
        
        while self.stream_buffer.qsize()>0:
            try:
                entry = self.stream_buffer.get()
                c_path = entry[0]
                c_img = entry[1]
                d_path = entry[2]
                d_img = entry[3]
                ir_path = entry[4]
                ir_img = entry[5]
                cv2.imwrite(c_path, c_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                cv2.imwrite(d_path, d_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                cv2.imwrite(ir_path, ir_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            except Exception as e:
                break

    
    def register_cam(self, cam_key, cam_object):
        """
        Add camera to the cam controller class to enable synchronized start
        """
        self.cams[cam_key] = cam_object
    
    def start(self):
        """
        Start the setup for the webcams and kinect
        """
        Thread(target=self.__start_webcams__, args=()).start()#start webcam setup
        self.__get__()#start kinect setup
        
        
    def __start_webcams__(self):
        #setup kinect. Must be done first so that kinect does not get recognized as webcam
        print(f"{self.debug_base}Waiting for Azure Kinect to start..")
        #start kinect thread
        while not self.kinect_setup_done:
            if self.ready_state:
                self.kinect_setup_done = True
        print(f"{self.debug_base}Azure Kinect ready state: True")
        #setup other webcams
        for entry in self.cams:
            self.cams[entry].start()
        
        self.check_ready_states() #wait until they can all read frames and start logging at the same time

    def check_ready_states(self):

        print(f"{self.debug_base}Waiting for all cams to start..")

        while not self.setup_done: #while not all webcams are ready, keep checking them every n seconds

            ready_states = [self.cams[x].ready_state for x in self.cams]
            debug_states = dict([(self.cams[x].name, self.cams[x].ready_state) for x in self.cams])
            self.setup_done = all(ready_states)

            time.sleep(5) #initializing cams takes time. Check every N seconds if they're ready and continue
            print(f"{self.debug_base}Camera ready state: {debug_states}")
            #check if everything's ready
            if self.setup_done:
                self.permission = True # allow kinect to start logging
               
                Thread(target=self.__write_log__, args=()).start()
                Thread(target=self.__write_img__, args=()).start()

                for entry in self.cams:
                    self.cams[entry].set_permission(True) #allow webcams to start logging
        #Thread(target=self.manage, args=()).start()
        print(f"{self.debug_base}All cameras ready. Logging has started.")
        print("Press CTRL+C while inside the console to quit the logging script.")

        


        #------depricated------
        # --> only for debugging, not necessary otherwise
        # for entry in self.cams:
        #     print(f"{self.debug_base}{self.cams[entry].name} currently with {self.cams[entry].get_fps()} fps at {self.cams[entry].get_size()}.")
        #self.manage()
        #------depricated------
    
    #depricated
    def manage(self):
        """
        Function that checks the current length of the buffers
        Was used in an earlier version to check if the cameras were still recording and how quickly the buffers are being emptied
        """
        while not self.stopped:
            #check cam buffer lengths
            buffer_lengths = [self.cams[x].stream_buffer.qsize() for x in self.cams]
            debug_lengths = dict([(self.cams[x].name, self.cams[x].stream_buffer.qsize()) for x in self.cams])
            print(f"{self.debug_base}Camera buffer lengths: {debug_lengths}")
            time.sleep(20)
    
    def cleanup(self):
        """
        Once logging is finished this function should be called to make the shutdown details visible
        """
        print("Writing remaining data to disk..")

        if self.debug:
            print(f"{self.debug_base}Processing Framerate logs..")
            #try:
            self.show_framerate_data()
            # except Exception as e:
            #     print(f"{self.debug_base}Plotting failed: {e}")
            
        print("Logging finished.")

    def show_framerate_data(self):
        """
        Function that shows the framerate data for all cameras and the kinect and calculates the mean and standard deviation
        """
        lists = [x.debug_frequency_log for x in self.cams.values()]
        lists.append(self.debug_frequency_log)
        for idx, lst in enumerate(lists):
            values = [t[1] for t in lst]
            mean = statistics.mean(values)
            stdev = statistics.stdev(values)
            cam_name = "Webcam"
            if idx == 5: cam_name = "Azure Kinect"
            print(f"{cam_name} {idx + 1}:")
            print(f"Mean: {mean}")
            print(f"Standard Deviation: {stdev}")
            print("===============================================")


    def shutdown(self):
        """
        If logging is supposed to be closed call this function
        """
        #stop azure kinect
        self.stopped = True
        #stop webcams
        for entry in self.cams:
            self.cams[entry].stop()
        Thread(target=self.cleanup(), args=()).start()

   
    

    



if __name__ == "__main__":
    if number_of_cameras <= 0:
        print(f"You need more than 0 cameras (You gave me {number_of_cameras})...because you want to record something ;)")
        sys.exit(-1)

    """ 
    # Not needed anymore. You have to enter a ID in the UI start screen
    if len(sys.argv) != 2:
        print('I need the participant ID...please ;)')
        sys.exit()
    """


    if not os.path.exists(video_base_path):
        os.makedirs(video_base_path)
        print('Generated new path ' + video_base_path)
    
    participant_id = "test_all"#sys.argv[1]
    img_full_path = video_base_path+participant_id+"/"
    log_full_path = log_base_path+participant_id+"/"
    controller = CamController(img_path=img_full_path, log_save_path=log_full_path) 

    #During the setup phase for all cams, the application can only be closed via ctrl+c
    try:
        for i in tqdm([0,1,3,4,5], "Creating camera objects..", colour="BLUE"): #the int indeces are the camera ids. It can happen that a device restart causes the ids to change. Make sure that the id that would call the kinect is not used for a webcam
            id = i
            cam = WebcamStream(id=id, img_save_path=img_full_path, log_save_path=log_full_path, fps=fps, frame_size=frame_size, buffer_size=buffer_size, debug=True, show=(id == visible_cam))
            controller.register_cam(cam.name, cam)
        print(f"Initializing cam controller..")
        controller.start()

    except KeyboardInterrupt: #check if ctrl+c is pressed during setup phase
        controller.shutdown()

    # if controller.setup_done:
        #     key = input("Press q and then enter to shutdown all cameras.")
        #     if key == "q": controller.shutdown()

    ########## DEBUGGING ###########
    #  if not self.joint_coords: self.log_buffer.put(self.last_line)
    #                 else: 
    #                     line = ";".join([str(datetime.now()), 
    #                                     str(self.ret_color), 
    #                                     str(self.ret_ir), 
    #                                     str(self.ret_depth), 
    #                                     str(self.c_name), 
    #                                     str(self.d_name), 
    #                                     str(self.ir_name)])+self.joint_coords+"\r"
    #                     self.log_buffer.put(line)
    #                     self.last_line = line
