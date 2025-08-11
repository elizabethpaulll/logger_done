import cv2
import multiprocessing
import csv
import time
import os
from datetime import datetime
from queue import SimpleQueue
import threading
from concurrent.futures import ThreadPoolExecutor

"""
Depricated version testing out multiprocessing for multiple cameras. An adapted version is used in the final implementation but this was kept to test out ideas without having to change the main logger structure.
"""


BUFFER_SIZE = 10  # Number of frames to buffer before writing

def frame_capture_process(webcam_id, frame_queue, terminate_signal):
    cap = cv2.VideoCapture(webcam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while not terminate_signal.value:
        ret, frame = cap.read()
        if ret:
            frame_queue.put(frame)

    cap.release()

def buffered_frame_saving_process_v2(frame_queue, img_save_path, log_save_path, webcam_id, fps=30, terminate_signal=None):
    frame_count = 0
    time_per_frame = 1.0 / fps
    frame_buffer = []
    csv_buffer = []

    csv_file = open(os.path.join(log_save_path, f"cam_{webcam_id}_log.csv"), "w", newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Timestamp", "Frame_Path"])
    
    # Function to save a batch of frames
    def save_frame_batch(frames):
        nonlocal frame_count
        for frame, timestamp in frames:
            frame_name = f"frame{frame_count}.png"
            frame_path = os.path.join(img_save_path, frame_name)
            cv2.imwrite(frame_path, frame)  # Saving in PNG format for now
            csv_buffer.append([timestamp, frame_path])
            frame_count += 1
        csv_writer.writerows(csv_buffer)
        csv_buffer.clear()

    while True:
        if terminate_signal.value and frame_queue.empty() and not frame_buffer:
            break

        if not frame_queue.empty():
            frame_data = frame_queue.get()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            frame_buffer.append((frame_data, timestamp))
            
            # If buffer is full, write frames to disk
            if len(frame_buffer) == BUFFER_SIZE:
                save_frame_batch(frame_buffer.copy())
                frame_buffer.clear()

        # Handle remaining frames in buffer if any
        if terminate_signal.value and not frame_queue.empty() and frame_buffer:
            save_frame_batch(frame_buffer.copy())
            frame_buffer.clear()
        
        time.sleep(time_per_frame)
    
    csv_file.close()

def parallel_webcam_process_multiprocessing_optimized_v3(webcam_id, img_save_path, log_save_path, fps=30, terminate_signal=None):
    try:
        frame_queue = multiprocessing.Queue(maxsize=20)  # Buffer for 20 frames

        capture_process = multiprocessing.Process(target=frame_capture_process, args=(webcam_id, frame_queue, terminate_signal))
        saving_process = multiprocessing.Process(target=buffered_frame_saving_process_v2, args=(frame_queue, img_save_path, log_save_path, webcam_id, fps, terminate_signal))
        
        capture_process.start()
        saving_process.start()
        
        capture_process.join()
        saving_process.join()

    except Exception as e:
        print(f"Error in webcam process for Webcam {webcam_id}: {str(e)}")

def main_with_termination():
    processes = []
    terminate_signal = multiprocessing.Value('b', False)  # boolean flag for termination

    try:
        for i in range(5):  # Assuming webcams have IDs 0 to 4
            img_save_path = f"dataset/images/cam_{i}"
            log_save_path = f"dataset/logs"
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)
            p = multiprocessing.Process(target=parallel_webcam_process_multiprocessing_optimized_v3, args=(i, img_save_path, log_save_path, 30, terminate_signal))
            p.start()
            processes.append(p)

        while True:
            all_finished = True
            for p in processes:
                p.join(timeout=1)  # Join with a timeout
                if p.is_alive():
                    all_finished = False

            if all_finished:
                break

            # If termination signal is set, break out of the loop
            if terminate_signal.value:
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                break

    except KeyboardInterrupt:
        print("Received keyboard interrupt. Signaling processes to terminate...")
        terminate_signal.value = True
        for p in processes:
            if p.is_alive():
                p.terminate()

if __name__ == "__main__":
    main_with_termination()
