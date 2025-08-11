import cv2
from multiprocessing import Process, Value, Queue
from threading import Thread
import csv
from tqdm import tqdm
import cv2
import pykinect_azure as pykinect
from utils import JOINTS, get_joint_coordinates, empty_line
from datetime import datetime
import ctypes
import numpy as np
import pandas as pd
import time
import os
import pickle


"""
This script orchestrates the logging and image recording for multiple cameras, including webcams and Azure Kinect. It ensures efficient data processing, storage, and performance evaluation through multiprocessing and multithreading.

Key Functions:

1. `log_consumer`:
   - Processes log entries from a queue and writes them to a CSV file.
   - Combines prefix data (timestamp and success flags) with skeleton data (e.g., joint coordinates).
   - Designed to run continuously, consuming entries from the queue until a termination signal is received.

2. `azure_img_consumer`:
   - Consumes image frames and log entries from queues for Azure Kinect data.
   - Writes synchronized video files for color, infrared (IR), and depth frames.
   - Logs data in a CSV file, ensuring synchronization between visual data and joint coordinates.

3. `webcam_producer`:
   - Captures frames from a webcam using OpenCV and writes them to a video file.
   - Logs frame success flags and timestamps to a CSV file.
   - Configured for specific resolution, frame rate, and codec to ensure consistent output.

4. `azure_producer`:
   - Captures frames from Azure Kinect, including color, IR, and depth data.
   - Extracts and logs joint coordinates using Azure Kinect's body tracking SDK.
   - Uses queues to decouple data capture from disk writing, improving performance.

5. `calculate_mean_log_rate` & `calculate_std`:
   - Analyze logging performance by calculating the mean rate and standard deviation of log entries per second.
   - Provide insights into the consistency and reliability of the data logging process.

Workflow:

1. **Initialization**:
   - Sets up directories for image and log file storage based on participant-specific paths.
   - Reads camera configurations from a serialized file (`pickle`) to identify connected devices.
   - Initializes shared memory variables and queues to enable inter-process communication.

2. **Producer Processes**:
   - Webcam producers capture frames and push them into local storage, logging their status and timestamps in parallel.
   - The Azure Kinect producer captures high-fidelity images (color, IR, and depth) and tracks body joints, pushing the data to separate queues for images and logs.

3. **Consumer Processes**:
   - Log consumers fetch data from log queues and write it into CSV files.
   - Image consumers fetch frames from the image queue and save them as MP4 video files using specified codecs and resolutions.
   - By separating production (capture) and consumption (writing), the system reduces bottlenecks and enhances throughput.

4. **Synchronization**:
   - Producers and consumers communicate through queues to ensure smooth data flow.
   - Termination signals (e.g., `None` in queues) are used to signal the end of data streams, allowing consumers to cleanly exit after processing all pending data.

5. **Performance Evaluation**:
   - After recording, the script analyzes log files to calculate the average and standard deviation of the logging rates.
   - This helps ensure the system performs reliably and highlights any inconsistencies during data capture.

6. **Graceful Shutdown**:
   - Waits for user input to signal the end of recording.
   - Ensures all producer and consumer processes complete their tasks and flush any remaining data before exiting.
"""



def log_consumer(log_queue, path, cam_index, header):
    """
    Consumes log entries from the queue and writes them to disk at specified path.
    """
    with open(path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        while True:
            try:
                entry = log_queue.get()
                if entry is None:
                    break
                prefix, skeleton = entry
                if skeleton is None:
                    prefix.extend(empty_line(32))
                    continue
                else: 
                    prefix.extend(skeleton)
            except Exception as e:
                print(f"azure log while schleife: {e}")
                continue
            csv_writer.writerow(prefix)
    
    print(f"Log consumer for camera {cam_index} complete.")


def azure_img_consumer(img_folder, log_folder, img_queue, log_queue, path, log_path, header, stopped):
    """
    Consumes images from the queue and writes them to disk as a video. Azure requires separate consumer due to preprocessing and image types.
    """
    img_path = img_folder+"azure/"
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(log_folder): os.makedirs(log_folder)
    log_path = log_folder+log_path

    c_out_path = img_path + path + "_color.mp4"
    ir_out_path = img_path + path + "_ir.mp4"    
    d_out_path = img_path + path + "_depth.mp4"
    c_writer = cv2.VideoWriter(c_out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1920, 1080))
    ir_writer = cv2.VideoWriter(ir_out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (512, 512), 0) 
    d_writer = cv2.VideoWriter(d_out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (512, 512), 0)
    finished = False
    try:
        with open(log_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(header)
            while not finished:
                for i in range(img_queue.qsize()):
                    try:
                        f = img_queue.get()
                        c_image, ir_image, d_image = f
                    except Exception:
                        pass
                    try:
                        entry = log_queue.get()
                        prefix, skeleton = entry
                        if skeleton is None:
                            prefix.extend(empty_line(32))
                            continue
                        else: 
                            prefix.extend(skeleton)
                    except Exception as e:
                        pass
                    if f is None and entry is None:
                            finished = True
                            break
                    if f:
                        try:
                            c_writer.write(c_image[:, :, :3])
                            ir_writer.write((ir_image).astype(np.uint8))
                            d_writer.write((d_image).astype(np.uint8))
                        except Exception:
                            #print(f"azure img while schleife: {e}")
                            continue
                    if entry:csv_writer.writerow(prefix)

                #print(f"img_queue size: {img_queue.qsize()}")
                #time.sleep(1)
        c_writer.release()
        ir_writer.release()
        d_writer.release()
        print(f"Recording complete for camera azure kinect.")
    except Exception:
        print("Error while writing azure kinect images. Azure Kinect Process terminated.")
        pass

def webcam_producer(cam_index, img_folder, log_folder, output_path, csv_path, stopped):
    """
    Produces images from the webcam and writes them to the queue.
    """
    img_path = img_folder+f"{cam_index}/"
    try:
        if not os.path.exists(img_path): os.makedirs(img_path)
        if not os.path.exists(log_folder): os.makedirs(log_folder)
    except Exception:
        pass 
    output_path = img_path+output_path
    csv_path = log_folder+csv_path

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    #ret, frame = cap.read()  # Read one frame to get the dimensions


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (1920, 1080))

    try:
        with open(csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Timestamp', f'cam{cam_index}_success'])

            while not stopped.value:
                ret, frame = cap.read()
                current_time = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
                if ret:
                    out.write(frame)
                csv_writer.writerow([current_time, ret])
            

        cap.release()
        out.release()
        print(f"Recording complete for camera {cam_index}")
    except Exception:
        print(f"Error while writing webcam {cam_index} images. Webcam {cam_index} Process terminated.")
        pass

def azure_producer(img_folder, log_folder, output_path, csv_path, stopped, img_queue, log_queue):
        header = ['Timestamp', 'c_success','ir_success', 'd_success']
        header.extend(list(JOINTS.keys()))


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
        #Thread(target=log_consumer, args=(log_queue, csv_path, 5, header, True)).start()
        Process(target=azure_img_consumer, args=(img_folder, log_folder ,img_queue, log_queue, output_path, csv_path, header, stopped)).start()
        # Start reading frames
        while not stopped.value:

            try:
                capture = device.update()
                body_frame = body_tracker.update()
            except Exception:
                continue

        
            ret_color, c_image = capture.get_color_image()#get color			
            ret_ir, ir_image = capture.get_ir_image()#get infrared
            ret_depth, d_image = capture.get_depth_image()#get depth
            if not ret_color or not ret_ir or not ret_depth:
                continue
            img_queue.put((c_image, ir_image, d_image))
            joint_coords = get_joint_coordinates(body_frame)#get joint coordinates
            timestamp = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
            log_queue.put(([timestamp, str(ret_color), str(ret_ir), str(ret_depth)], joint_coords))

        log_queue.put(None) 
        img_queue.put(None)

def calculate_mean_log_rate(logfile_path):
    """
    Calculate the mean rate of entries per second for a given logfile.
    """
    df = pd.read_csv(logfile_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    start_time = df['Timestamp'].min()
    end_time = df['Timestamp'].max()
    total_duration_seconds = (end_time - start_time).total_seconds()
    total_entries = len(df)

    return total_entries / total_duration_seconds

def calculate_std(logfile_path):
    """
    Calculate the standard deviation of the rate of entries per second for a given logfile.
    """
    df = pd.read_csv(logfile_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Group by each second and count the number of entries
    df['Second'] = df['Timestamp'].dt.floor('S')
    entries_per_second = df.groupby('Second').size()

    # Calculate standard deviation
    std_rate = entries_per_second.std()

    return std_rate

if __name__ == "__main__":
    # output_files = ['webcam_1.avi', 'webcam_2.avi', 'webcam_3.avi', 'webcam_4.avi', 'webcam_5.avi']
    # csv_files = ['webcam_1.csv', 'webcam_2.csv', 'webcam_3.csv', 'webcam_4.csv', 'webcam_5.csv']
    video_base_path = 'dataset/images/'
    log_base_path = 'dataset/logs/'
    
    p_id = input("Enter participant ID: ")
    #cams = input("Enter the number of cameras to record from (including depth sensors):")
    img_full_path = video_base_path+p_id+"/"
    log_full_path = log_base_path+p_id+"/"
    
    stopped = Value(ctypes.c_bool, False)

    stream_queue = Queue()
    log_queue = Queue()

    processes = []
    cams = []
    with open('cam_config.pickle', 'rb') as f:
        cam_dict = pickle.load(f)
        for i in cam_dict.keys():
            if cam_dict[i] == 'Azure Kinect 4K Camera':
                continue
            cams.append(int(i))
    for i in tqdm(cams, "Creating camera objects..", colour="green"):
        p = Process(target=webcam_producer, args=(i, img_full_path, log_full_path, f"webcam_{i}.mp4", f"webcam_{i}.csv", stopped))
        p.start()
        processes.append(p)
        time.sleep(1)
    
    print("Creating azure kinect object..")
    time.sleep(1)
    azure_p = Process(target=azure_producer, args=(img_full_path, log_full_path, f"webcam_azure_kinect", f"webcam_azure_kinect.csv", stopped, stream_queue, log_queue))
    azure_p.start()
    print("Starting recording..")
    processes.append(azure_p)

    command = input("Press Enter to stop recording.")
    stopped.value = True
    for p in processes:
            p.join()
    print("Evaluating logging performance..")
    logfile_paths = {}
    for i in cams:
        logfile_paths[f'webcam_{i}'] = f'{log_full_path}webcam_{i}.csv'
        # f'webcam_{cams[1]}': f'{log_full_path}/webcam_{cams[1]}.csv',
        # f'webcam_{cams[2]}': f'{log_full_path}/webcam_{cams[2]}.csv',
        # f'webcam_{cams[3]}': f'{log_full_path}/webcam_{cams[3]}.csv',
        #f'webcam_{cams[4]}': f'webcam_{cams[4]}.csv',
        #'webcam_6': f'webcam_{cams[5]}.csv',
    logfile_paths['webcam_azure_kinect'] = f'{log_full_path}webcam_azure_kinect.csv'

    # Calculate and print the mean rate for each logfile
    for cam, path in logfile_paths.items():
        mean_rate = calculate_mean_log_rate(path)
        std  = calculate_std(path)
        print(f"{cam}: M = {mean_rate:.2f} hz (SD={std:.2f})")

    # Automatically merge gesture labels with camera logs
    try:
        from merge_gesture_labels import merge_gesture_labels_for_pid
        merge_gesture_labels_for_pid(p_id)
        print("Automatic merging of gesture labels complete.")
    except Exception as e:
        print(f"Automatic merging failed: {e}")
