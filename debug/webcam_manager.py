
from webcam_stream import WebcamStream
from azure_stream import AzureKinectStream
import time
import sys
import os
from tqdm import tqdm

"""
Depricated of the camcontroller class. This version is not used in the final implementation of the project but was kept to test out additional capturing methods to deal with multiple cameras at the same time.
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


class CamController:
    def __init__(self) -> None:
        self.cams = {}
        self.kinect = None
        self.setup_done = False
        self.kinect_setup_done = False
        self.debug_base = "[CAM CONTROLLER]: "
        self.path = log_base_path+ "log_cam_all__participant " + participant_id + ".csv"
        # self.camera_log_file = open(log_base_path+ "log_cam_all__participant " + participant_id + ".csv", "w")
        # if os.path.isfile(self.path): self.camera_log_file = open(log_base_path+ "log_cam_all__participant " + participant_id + ".csv", "a")

    
    def register_cam(self, cam_key, cam_object):
        self.cams[cam_key] = cam_object
    
    def register_kinect(self, kinect_object):
        self.kinect = kinect_object
    
    def start_kinect(self):
        self.kinect.start()
        print(f"{self.debug_base}Waiting for Azure Kinect to start..")
        while not self.kinect_setup_done:
            if self.kinect.ready_state:
                self.kinect_setup_done = True
        print(f"{self.debug_base}Azure Kinect ready state: True")

    def start_webcams(self):
        for entry in self.cams:
            self.cams[entry].start()
        
        self.check_ready_states()

    def check_ready_states(self):

        print(f"{self.debug_base}Waiting for all cams to start..")

        while not self.setup_done:
            ready_states = [self.cams[x].ready_state for x in self.cams]
            debug_states = dict([(self.cams[x].name, self.cams[x].ready_state) for x in self.cams])
            self.setup_done = all(ready_states)
            time.sleep(2)#initializing cams takes time. Check every N seconds if they're ready and continue
            print(f"{self.debug_base}Camera ready state: {debug_states}")
            if self.setup_done:
                self.kinect.set_permission(True)
                for entry in self.cams:
                    self.cams[entry].set_permission(True)

        print(f"{self.debug_base}All cameras ready. Starting.")
        # for entry in self.cams:
        #     print(f"{self.debug_base}{self.cams[entry].name} currently with {self.cams[entry].get_fps()} fps at {self.cams[entry].get_size()}.")
        self.manage()
    
    def manage(self):
        while True:
            #check if cams failed
            failures = [(self.cams[x].stopped) for x in self.cams]

            if all(failures):
                print(f"{self.debug_base}All cams disconnected. Shutting down.")
                self.shutdown()

    def shutdown(self):
        #stop webcams
        for entry in self.cams:
            self.cams[entry].stop()
        #stop azure kinect
        self.kinect.stop()


if __name__ == "__main__":

    if number_of_cameras <= 0:
        print(f"You need more than 0 cameras (You gave me {number_of_cameras})...because you want to record something ;)")
        sys.exit(-1)

    # if len(sys.argv) != 2:
    #     print('I need the participant ID...please ;)')
    #     sys.exit()
    
    if not os.path.exists(video_base_path):
        os.makedirs(video_base_path)
        print('Generated new path ' + video_base_path)
    
    participant_id = "test_all"#sys.argv[1]
    controller = CamController() 

    #During the setup phase for all cams, the application can only be closed via ctrl+c
    try:
        img_full_path = video_base_path+participant_id+"/"
        log_full_path = log_base_path+participant_id+"/"
        kinect = AzureKinectStream(log_full_path, img_full_path, fps=30, debug=True)
        controller.register_kinect(kinect)
        controller.start_kinect()
        for i in [0,1,3,4,5]:
            id = i
            cam = WebcamStream(id=id, img_save_path=img_full_path, log_save_path=log_full_path, fps=fps, frame_size=frame_size, buffer_size=buffer_size, debug=True, show=(id == visible_cam))
            print(id)
            controller.register_cam(cam.name, cam)
        print(f"Starting {number_of_cameras} webcams.")
        print("Press CTRL+C while inside the console to quit the webcam script.")
        controller.start_webcams()

    except KeyboardInterrupt: #check if ctrl+c is pressed during setup phase
        controller.shutdown()
    

    if controller.setup_done:
        key = input("Press q and then enter to shutdown all cameras.")
        if key == "q": controller.shutdown()

