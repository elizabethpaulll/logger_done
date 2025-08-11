#necessary libs
import cv2
import numpy as np

'''
This script is meant for debugging only. The way it was used previously was to check if all the cameras are working properly and if they are connected to the correct ports.
Further, when setting up the cameras in a new environment (i.e., driving simulator, new room, car, etc.), this script can be used to check all the angles and make sure that the cameras are placed correctly.
'''

# Function to resize and pad the frames received from webcams
def resize_and_pad_frame(frame, target_width=640, target_height=480):
    """Resizes the frame while maintaining aspect ratio and pads to target size."""
    # Calculate the scale factor and resize
    height, width = frame.shape[:2]
    scale = min(target_width/width, target_height/height)
    resized_frame = cv2.resize(frame, (int(width*scale), int(height*scale)))

    # Calculate padding sizes
    delta_w = target_width - resized_frame.shape[1]
    delta_h = target_height - resized_frame.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    # Add padding to the resized image
    color = [0, 0, 0]  # Black padding
    padded_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded_frame

def create_grid(frames, grid_size=(2, 3)):
    """Creates a grid layout of the frames, resizing them to a common size."""
    if not frames:
        return None

    # Resize and pad frames to a common size
    processed_frames = [resize_and_pad_frame(frame) for frame in frames]

    # Create rows for the grid
    rows = []
    for i in range(0, len(processed_frames), grid_size[1]):
        row_frames = processed_frames[i:i+grid_size[1]]
        while len(row_frames) < grid_size[1]:  # Fill in missing frames if any
            row_frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
        rows.append(np.hstack(row_frames))

    # Combine rows to create the grid
    return np.vstack(rows)

# Initialize cameras with specified resolution
camera_config = {0: 'Azure Kinect 4K Camera',
                 1: 'HD Pro Webcam C920',
                 2: 'HD Pro Webcam C920',
                 3: 'HD Pro Webcam C920',
                 4: 'HD Pro Webcam C920',
                 5: 'HD Pro Webcam C920'}
cameras = []
for id in camera_config.keys():
    cap = cv2.VideoCapture(id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cameras.append(cap)

try:
    while True:
        frames = []
        labels = []
        for id, cap in zip(camera_config.keys(), cameras):
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, camera_config[id], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frames.append(frame)

        grid_frame = create_grid(frames, grid_size=(2, 3))
        if grid_frame is not None:
            cv2.imshow('Camera Grid', grid_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted by user")

# Release resources
for cap in cameras:
    cap.release()
cv2.destroyAllWindows()