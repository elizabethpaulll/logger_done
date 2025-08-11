import cv2
import numpy as np
import pykinect_azure as pykinect
import csv
import os
import time
from utils import get_joint_coordinates
from threading import Thread

def busy_wait():
     start_time = time.time()
     while time.time()-start_time<=10:
          pass
     
def test_azure_frame_performance(id):
        # Initialize the library
        pykinect.initialize_libraries(track_body=True)

        # Modify camera configuration
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        device_config.color_format = pykinect.K4A_FRAMES_PER_SECOND_30
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

        # Start device
        device = pykinect.start_device(config=device_config)

        # Start body tracker
        bodyTracker = pykinect.start_body_tracker()

        # Open CSV file in write mode
        #with open('joint_coordinates.csv', 'w', newline='') as csvfile:
            #csv_writer = csv.writer(csvfile)

            # Update the header to include paths to captures and joint names
            #csv_writer.writerow(["timestamp", "color_image_path", "depth_image_path", "ir_image_path"] + joint_names)

        tics = []
        i = 0
        start_time = time.time()
        limit = 10
        while (time.time()-start_time)<=limit:
            tic = time.time()
            
            # Get capture
            capture = device.update()

            # Get body tracker frame
            body_frame = bodyTracker.update()

            # Get the color image
            ret_color, ir_image = capture.get_ir_image()
            ret_color, color_image = capture.get_color_image()
            ret_color, depth_image = capture.get_depth_image()
            get_joint_coordinates(body_frame)

            r = time.time()-tic
            tics.append(r)
            #print(r)
            i+=1
        end_time = time.time()
        
        print(f"{id}_Avg_log_frequency:{sum(tics)/len(tics)}")
        print(f"{len(tics)} in {limit} seconds)")
            

            # Save images and get their paths
            #timestamp_str = str(int(time.time()*1000))
            # color_path = os.path.join(color_dir, f"color_{timestamp_str}.jpg")
            # depth_path = os.path.join(depth_dir, f"depth_{timestamp_str}.jpg")
            # ir_path = os.path.join(ir_dir, f"ir_{timestamp_str}.jpg")
            # print(f"{timestamp_str}: Data received.")

            # min_val = np.amin(depth_image)
            # max_val = np.amax(depth_image)

            # all_min.append(min_val)
            # all_max.append(max_val)
            
            # cv2.imwrite(color_path, color_image)
            # cv2.imwrite(depth_path, depth_image)
            # cv2.imwrite(ir_path, ir_image)
            # Write the paths and joint coordinates to the CSV
            #csv_writer.writerow([timestamp_str, color_path, depth_path, ir_path] + joints_3d)
        
        # print(f"Min: {np.amin(np.array(all_min))}")
        # print(f"Max: {np.amax(np.array(all_max))}")


if __name__ == "__main__":
    #Thread(target=test_azure_frame_performance, args=([0])).start()
    test_azure_frame_performance(1)
    # for i in range(20):
    #      Thread(target=busy_wait, args=()).start()
