import cv2
import numpy as np
import pykinect_azure as pykinect

"""
These are helper functions for the PyKinect module. They are used to extract joint information from the body frame. Check this file in case errors with the way joints are accessed arise.
"""

#dictionary of joints names as defined in the C++ API for quick access
JOINTS = {
    "neck": pykinect.K4ABT_JOINT_NECK,
    "nose": pykinect.K4ABT_JOINT_NOSE,
    "pelvis": pykinect.K4ABT_JOINT_PELVIS,
    "wrist-left": pykinect.K4ABT_JOINT_WRIST_LEFT,
    "wrist-right": pykinect.K4ABT_JOINT_WRIST_RIGHT,
    "elbow-left": pykinect.K4ABT_JOINT_ELBOW_LEFT,
    "elbow-right": pykinect.K4ABT_JOINT_ELBOW_RIGHT,
    "thumb-left": pykinect.K4ABT_JOINT_THUMB_LEFT,
    "thumb-right": pykinect.K4ABT_JOINT_THUMB_RIGHT,
    "ear-left": pykinect.K4ABT_JOINT_EAR_LEFT,
    "ear-right": pykinect.K4ABT_JOINT_EAR_RIGHT,
    "head": pykinect.K4ABT_JOINT_HEAD,
    "clavicle-left": pykinect.K4ABT_JOINT_CLAVICLE_LEFT,
    "clavicle-right": pykinect.K4ABT_JOINT_CLAVICLE_RIGHT,
    "eye-left": pykinect.K4ABT_JOINT_EYE_LEFT,
    "eye-right": pykinect.K4ABT_JOINT_EYE_RIGHT,
    "hand-left": pykinect.K4ABT_JOINT_HAND_LEFT,
    "hand-right": pykinect.K4ABT_JOINT_HAND_RIGHT,
    "handtip-left": pykinect.K4ABT_JOINT_HANDTIP_LEFT,
    "handtip-right": pykinect.K4ABT_JOINT_HANDTIP_RIGHT,
    "foot-left": pykinect.K4ABT_JOINT_FOOT_LEFT,
    "foot-right": pykinect.K4ABT_JOINT_FOOT_RIGHT,
    "ankle-right": pykinect.K4ABT_JOINT_ANKLE_RIGHT,
    "ankle-left": pykinect.K4ABT_JOINT_ANKLE_LEFT,
    "hip-left": pykinect.K4ABT_JOINT_HIP_LEFT,
    "hip-right": pykinect.K4ABT_JOINT_HIP_RIGHT,
    "shoulder-left": pykinect.K4ABT_JOINT_SHOULDER_LEFT,
    "shoulder-right": pykinect.K4ABT_JOINT_SHOULDER_RIGHT,
    "spine-chest": pykinect.K4ABT_JOINT_SPINE_CHEST,
    "spine-navel": pykinect.K4ABT_JOINT_SPINE_NAVEL,
    "knee-left": pykinect.K4ABT_JOINT_KNEE_LEFT,
    "knee-right": pykinect.K4ABT_JOINT_KNEE_RIGHT,
}

#dictionary of confidence levels as defined in the C++ API in case int levels should be replaced with CL-names
CONFIDENCE = {0: 'none', 1: 'low', 2: 'medium', 3: 'high'}

#quick access list that can be used to iterate over all joints
joint_names = keys_list = list(JOINTS.keys())

#function that only gets the coordinates of the joints
def get_joint_coordinates(body_frame):
    """This function still requires some bug handeling. If no people are in the camera frame
    accessing the body frame will throw an error. This is not handeled yet.

    Args:
        body_frame (object): Kinect BodyFrame object

    Returns:
        list (string) : A string list containing 4 dim. vectors of the joint coordinates. Idx 0-2 are the x,y,z coordinates, idx 3 is the confidence level
    """
    try:
        body_id = 0
        skeleton_3d = body_frame.get_body(body_id).numpy()
        joint_params = [str(skeleton_3d[JOINTS[x], [0,1,2,7]]) for x in joint_names]
        return joint_params
    except:
        return None

#function that builds the string of coordinates to be written to the csv file
def format_coordinates(params):
    if not params: return None
    else: return ";"+";".join(params)

#function that gets the joint coordinates, joint orientations, and confidence levels
def get_joint_information(body_frame):
    """This function still requires some bug handeling. If no people are in the camera frame
    accessing the body frame will throw an error. This is not handeled yet.

    Args:
        body_frame (object): Kinect BodyFrame object

    Returns:
        list (string): A string list containing vectors with 8 values of the joint coordinates. Idx 0-2 are the x,y,z coordinates, 3-7 are the orientations, 8 is the confidence level
    """
    body_id = 0
    skeleton_3d = body_frame.get_body(body_id).numpy()
    joint_params = [(skeleton_3d[JOINTS[x],:3], skeleton_3d[JOINTS[x],3:7], CONFIDENCE[skeleton_3d[JOINTS[x],7]]) for x in joint_names]
    #print(joint_params[0][0].dtype(), joint_params[0][1].dtype(), joint_params[0][2].dtype())
    return joint_params
    # except:
    #     print("No body in frame. Using empty coordinates. Step back into the frame brother.")
    #     return "Failure"

def empty_line(length):
    return ["" for x in range(length)]