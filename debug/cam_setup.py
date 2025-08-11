from pygrabber.dshow_graph import FilterGraph
from tqdm import tqdm
import pickle
import time
import cv2

'''
This script is meant for debugging only. When setting up multiple cameras on one PC, it's benefitial to know the ports of each cam as well as which cameras have been recognized correctly.
This script will help you find the camera ids of the cameras you want to use (uncomment the main function to use it).
'''

def cam_finder():
    #A function that keeps asking the user to supply camera ids until the user wants to quit
    #Does not return anything

    cam_ids = []
    print("Please enter the camera ids you want to use. Enter 'q' to quit.")
    while True:
        cam_id = input("Camera id to search: ")
        cam = cv2.VideoCapture(int(cam_id), cv2.CAP_DSHOW)
        end = input("End? (y/n): ")
        if end == 'y':
            cam.release
            break
        cam.release()



print("Camera setup started...")
def get_available_cameras() :

    devices = FilterGraph().get_input_devices()

    available_cameras = {}

    for i in tqdm(range(len(devices)), colour='green', desc='Checking device graph..'):
        available_cameras[i] = devices[i]
        time.sleep(1)

    return available_cameras



if __name__ == "__main__" :

    ################### Automatic camera setup ###################
    cam_dict = get_available_cameras()

    print(f"Found the following devices: {cam_dict}")

    print("Writing config file...")
    with open('cam_config.pickle', 'wb') as f:
        pickle.dump(cam_dict, f)

    print("Setup complete.")

    ################### Manual camera setup ###################
    cam_finder()
    
    
 


