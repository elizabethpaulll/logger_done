import pandas as pd
import glob
import os

def merge_gesture_labels_for_pid(pid):
    # Paths
    gesture_log_path = f'logs/auto_labels_{pid}.csv'
    cam_log_dir = f'dataset/logs/{pid}/'
    output_dir = cam_log_dir

    # Load gesture log
    gesture_log = pd.read_csv(gesture_log_path)
    gesture_log['Timestamp'] = pd.to_datetime(gesture_log['Timestamp'])

    # Find all camera logs for this participant
    cam_logs = glob.glob(os.path.join(cam_log_dir, 'webcam_*.csv'))

    for cam_log_path in cam_logs:
        # Skip already labeled logs
        if cam_log_path.endswith('_labeled.csv'):
            continue
        cam_log = pd.read_csv(cam_log_path)
        cam_log['Timestamp'] = pd.to_datetime(cam_log['Timestamp'])
        # Assign gesture label to each frame
        gesture_idx = 0
        labels = []
        for frame_time in cam_log['Timestamp']:
            # Move to the most recent gesture event before this frame
            while (gesture_idx + 1 < len(gesture_log) and
                   gesture_log['Timestamp'][gesture_idx + 1] <= frame_time):
                gesture_idx += 1
            if gesture_idx < len(gesture_log):
                labels.append(gesture_log['Gesture'][gesture_idx])
            else:
                labels.append('none')
        cam_log['Gesture'] = labels
        # Output labeled log
        out_path = cam_log_path.replace('.csv', '_labeled.csv')
        cam_log.to_csv(out_path, index=False)
        print(f"Labeled log written: {out_path}")

if __name__ == "__main__":
    # Get participant ID
    pid = input("Enter participant ID: ")
    merge_gesture_labels_for_pid(pid)
    print("Merging complete.") 