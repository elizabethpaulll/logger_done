### [MULTICAMERA LOGGER - DOCUMENTATION]
Author: Tim Fabian + Albin Zeqiri  

## Hardware
- Azure Kinect DK
- 6 Camera Views (5x Logitech + 1 AzureKinect)
    - Tested with Logitech C270 /C920
    - Should work with any cheap webcam

## Programming languages
- HTML, CSS, JAVASCRIPT
- PYTHON (VERSION 3.10, ctypes and Azure packages are currently causing issues with newer versions of Python) 



## Body tracking

| ID | Joint | Parent |
|:-----:|:----:|:------:|
|0  |PELVIS                    |-|
|1  |SPINE_NAVEL               |PELVIS|
|2  |SPINE_CHEST               |SPINE_NAVEL|
|3  |NECK                      |SPINE_CHEST|
|4  |CLAVICLE_LEFT             |SPINE_CHEST|
|5  |SHOULDER_LEFT             |CLAVICLE_LEFT|
|6  |ELBOW_LEFT                |SHOULDER_LEFT|
|7  |WRIST_LEFT                |ELBOW_LEFT|
|8  |HAND_LEFT                 |WRIST_LEFT|
|9  |HANDTIP_LEFT              |HAND_LEFT|
|10 |THUMB_LEFT                |WRIST_LEFT|
|11 |CLAVICLE_RIGHT            |SPINE_CHEST|
|12 |SHOULDER_RIGHT            |CLAVICLE_RIGHT|
|13 |ELBOW_RIGHT               |SHOULDER_RIGHT|
|14 |WRIST_RIGHT               |ELBOW_RIGHT|
|15 |HAND_RIGHT                |WRIST_RIGHT|
|16 |HANDTIP_RIGHT             |HAND_RIGHT|
|17 |THUMB_RIGHT               |WRIST_RIGHT|
|18 |HIP_LEFT                  |PELVIS|
|19 |KNEE_LEFT                 |HIP_LEFT|
|20 |ANKLE_LEFT                |KNEE_LEFT|
|21 |FOOT_LEFT                 |ANKLE_LEFT|
|22 |HIP_RIGHT                 |PELVIS|
|23 |KNEE_RIGHT                |HIP_RIGHT|
|24 |ANKLE_RIGHT               |KNEE_RIGHT|
|25 |FOOT_RIGHT                |ANKLE_RIGHT|
|26 |HEAD                      |NECK|
|27 |NOSE                      |HEAD|
|28 |EYE_LEFT                  |HEAD|
|29 |EAR_LEFT                  |HEAD|
|30 |EYE_RIGHT                 |HEAD|
|31 |EAR_RIGHT                 |HEAD|

Table from [Microsoft](https://learn.microsoft.com/de-de/azure/kinect-dk/body-joints)

### PyKinect-Version

A custom Azure Kinect stream was implemented to integrate the Azure Kinect. To do so, we based our implementation on the existing pykinect_azure repository by ibaiGarodo
Refer to their (in-depth) instruction for installation instructions: https://github.com/ibaiGorordo/pyKinectAzure.

#### Known Bugs:

- If no person is in the frame, the Kinect's skeleton model throws an error that does not stop the logger, but the Azure Kinect will not receive new data as the error originates in the Azure Kinect's own code. Should such a case occur, restart the logging.
- If the PC is restarted or something is reconnected, the IDs of the individual cameras must be checked, as they can be changed by Windows. It must be ensured that the ID that would address the Kinect is not given to a WebcamStream class, as this would block Azure from recording depth, infrared, and skeleton data.


#### UI
In Core, there is a subfolder named UI. Inside it, you will find:

UI  
|_Images  
&nbsp; &nbsp; &nbsp;   |_various backgrounds  
|_Function.js  
|_Labeling.html  
|_StartScreen.html  
|_Style.css  

Open StartScreen.html.

#### Getting the Logger to run
* First, ensure that all Logitech cameras are visible in the LogiTune tool (if not installed, download it, this is only relevant for Logitech C920 and better)
* The logging is supposed to be done with 6 cameras (5 webcams + 1 azure). If fewer cameras than that were detected, find a port distribution that does recognize all 6 cameras (NOTE: Azure requires USB 3.0)
* The camera views should be opened in Logitech, and lighting conditions adjusted to your preferences, as AutoExposure and Low Light Adaptation must be off for 30Hz recordings (this can be achieved by briefly turning Auto Exposure and Low Light Adaptation on and then off again or manually via the sliders)
* When all cameras are present, run cam_setup.py. If a camera is missing --> replug everything again or check if the Kinect is not connected to a USB 3.0 port
* If cam_setup.py finds 6 cameras (5 Logitech + 1 Kinect), you can start the recording
* To do this 1. Start the StartScreen.html to start the labeling interface, enter the PID and 2. run ``multi_processing_main.py``, enter the Participant ID there as well (both parts work independently, meaning if the labeling UI is closed, the logger must still be stopped in the console by pressing 'Enter'). More detailed explanations of code are included in the relevant files.
* During logging, the multi_processing_main.py will create multiple recordings and monitoring logs, where afterward, it can be verified if any frames got lost or corrupted due to throughput issues on your PC. The Labeling interface will create its own log file solely containing the timestamps when a gesture was started and when it ended. In the end, the frame logs from the video streams and the labeling interface must be merged manually

