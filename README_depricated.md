### [DOCUMENTATION DEPRICATED - WILL BE UPDATED SHORTLY]
# Azure Kinect
Author: Tim Fabian  
Datum: 10.07.2023

## Hardware
- Azure Kinect
- Webcams
    - Funktioniert nicht mit Logitech (270, 920)
    - Funktioniert mit 
    [BENEW YFull HD 1080P Webcam](https://www.amazon.de/BENEWY-Mikrofon-Webkamera-Studieren-Konferenzen/dp/B089LTY8X7/ref=sr_1_20?__mk_de_DE=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=30ES87EKIXTA0&keywords=hd+webcam&qid=1689071482&sprefix=hd+webcam+%2Caps%2C83&sr=8-20), 
    [Microsoft Life](https://www.amazon.de/Microsoft-LifeCam-HD-3000-Webcam-zertifiziert/dp/B0099XD1PU/ref=sr_1_3?keywords=webcam+microsoft&qid=1689076537&s=computers&sprefix=webcam+micr%2Ccomputers%2C74&sr=1-3)

## Programmiersprachen
- C#
- Python

## Software
Das Projekt besteht aus mehreren (Software-)Teilen, die unter [Core](./Core) zu finden sind. 


## Hinweis vorweg
Beim Ausführen der Software ist es extrem wichtig, dass erst RecordingWebcams und dann AzureKinect3 ausgeführt wird und die Azure Kinect erst nach dem erfolgreichen Starten von RecordingWebcams an den PC angeschlossen wird. Warum? Die Kinect wird sonst ggf. als Kamera annerkannt und von RecordingWebcams verwaltet, was hier nichts bringt. 
1. Webcams anschließen
2. RecordingWebcams starten und warten, bis das Bild der Kamera(s) erscheint (mindestens eins sollte erscheinen)
3. Kinect anschließen
4. AzureKinect3 starten

### Anschluss der Kameras an den USB Ports
<img src = "/Stuff/Kameras.jpg" width=200 height = 400/>
Die Kameras müssen richtig angeschlossen werden. Dabei dürfen keine zwei am selebn USB Controller hängen. 


### Recording Webcams
#### Funktion
[Recording Webcams](./Core/RecordingWebcams) zeichnet eine (im Script) festgelegte Anzahl an Kameras auf und loggt Zeit und Status (True = Kamera funktioniert; False = Fehler trat auf) der einzelnen Kameras. Die Verzeichnisse, in denen die Daten gespeichert werden, müssen im Skript (Abschnitt "Part of definitions") angegeben werden.  

#### Festzulegende Variablen
- pathVideos  (Path of direction where the videos should be saved)
- pathLogs (Path of direction where the logs should be saved) 
- number_of_cameras (number of cameras)

#### Aufruf 
Das Skript wird einfach mit `python RecordingWebcams.py <PID>` aufgerufen, wobei PID die ProbandenID ist. Ob Zahl oder Buchstaben...ist egal. Es wird allerdings nicht auf Eindeutigkeit geprüft!

#### Während der Ausführung
Direkt zu Beginn der Ausführung sollten in der Konsole Ausgaben wie "Thread 0" gestartet kommen. Dies sollte für Thread 0 bis "number_of_cameras" -1 passieren. Danach kann es sein, dass eine Weile nichts passiert (bis zu 30 Sekunden). Die Webcams sind ziemlich
langsam beim Starten...warum auch immer. Sobald die Webcams aufzeichen erscheinen Ausgaben wie "Camera 0 is writing" (ebenfalls von 0 bis "number_of_cameras" -1) und Camera 0 zeigt die aktuelle Ansicht an.  
Zum Beenden der Anwendung einfach das Fenster von Kamera 0 auswählen und "ESC" drücken. Dann werden alle Kameras und Threads nach und nach beendet.  
NIEMALS das Programm  anders beenden, da die Video Streams sonst nicht geschlossen werden und die Videos damit unbrauchbar sind. 

### Azure Kinect 2
Ein Projekt, das leider nicht so ganz funktioniert, aber theoretisch die Videos oder Kinect aufnehmen soll. Bodytracking und Logging funktioniert. Das Separieren der Videotracks leider nicht.

### Azure Kinect 3
#### Funktion
Das Program dient der Verwaltung, Loggen und Aufnahme der Kinect (in Bildern).

#### Festzulegende Variablen
- filePath (Pfad für Speicherort des Loggers)
- imagePath (Pfad für den Speicherort der Bilder der Kinect)
    - Es müssen hier die Unterordner "Color" und "Depth" existieren
- pID (ProbandenID)
    - Integer oder String
- timeShift (Zeitverschiebung relativ zu UTC -> Sollte somit 1 (Winterzeit) oder 2 (Sommerzeit) sein)

#### Ausführung
Das Skript wird bestenfalls über Visual Studio 2019 ausgeführt. Es sollte sich dann ein Control Panel öffnen, in dem Labels (19 verschiedene) per Klick ausgewählt werden können (grüner Button = aktives Label). Über "Stop Recording" kann das Program beendet werden. Bitte niemals killen, da der Logger und die Filewriter dann evtl. Blödsinn machen. 
Es ist möglich ein Label auszuwählen und durch erneutes Klicken wieder abzuwählen. Man kann aber ebenso ein anderes Label auswählen, wodurch das alte Label abgewählt und das neue ausgewählt wird (auch im Logger).  
Über dem Button "Stop Recording" steht zusätzlich der Name des aktuell ausgewählten Labels. Mit dem Text kann allerdings nicht interagiert werden ;)

<img src = "/Stuff/AzureKinect.png" width=400 height = 200/>
<img src = "/Stuff/KinectPanel.png" width=400 height = 200/>


## Body tracking
![Semantic description of image](/Stuff/joint-hierarchy.png "Gelenkpunkte beim Body tracking")

| ID | Gelenk | Übergeornetes Gelenk |
|:-----:|:----:|:------:|
|0  |PELVIS (BECKEN)                    |-|
|1  |SPINE_NAVEL (LENDENWIRBELSÄULE)    |PELVIS (BECKEN)|
|2  |SPINE_CHEST (BRUSTWIRBELSÄULE)     |SPINE_NAVEL (LENDENWIRBELSÄULE)|
|3  |NECK (HALS)                        |SPINE_CHEST (BRUSTWIRBELSÄULE)|
|4  |CLAVICLE_LEFT (SCHLÜSSELBEIN_LINKS)|SPINE_CHEST (BRUSTWIRBELSÄULE)|
|5  |SHOULDER_LEFT (SCHULTER_LINKS)     |CLAVICLE_LEFT (SCHLÜSSELBEIN_LINKS)|
|6  |ELBOW_LEFT (ELLENBOGEN_LINKS)      |SHOULDER_LEFT (SCHULTER_LINKS)|
|7  |WRIST_LEFT (HANDGELENK_LINKS)      |ELBOW_LEFT (ELLENBOGEN_LINKS)|
|8  |HAND_LEFT (HAND_LINKS)             |WRIST_LEFT (HANDGELENK_LINKS)|
|9  |HANDTIP_LEFT (FINGERSPITZE_LINKS)  |HAND_LEFT (HAND_LINKS)|
|10 |THUMB_LEFT (DAUMEN_LINKS)          |WRIST_LEFT (HANDGELENK_LINKS)|
|11 |CLAVICLE_RIGHT (SCHLÜSSELBEIN_RECHTS)|SPINE_CHEST (BRUSTWIRBELSÄULE)|
|12 |SHOULDER_RIGHT (SCHULTER_RECHTS)   |CLAVICLE_RIGHT (SCHLÜSSELBEIN_RECHTS)|
|13 |ELBOW_RIGHT (ELLENBOGEN_RECHTS)    |SHOULDER_RIGHT (SCHULTER_RECHTS)|
|14 |WRIST_RIGHT (HANDGELENK_RECHTS)    |ELBOW_RIGHT (ELLENBOGEN_RECHTS)|
|15 |HAND_RIGHT (HAND_RECHTS)           |WRIST_RIGHT (HANDGELENK_RECHTS)|
|16 |HANDTIP_RIGHT (FINGERSPITZE_RECHTS)|HAND_RIGHT (HAND_RECHTS)|
|17 |THUMB_RIGHT (DAUMEN_RECHTS)        |WRIST_RIGHT (HANDGELENK_RECHTS)|
|18 |HIP_LEFT (HÜFTE_LINKS)|PELVIS (BECKEN)|
|19|KNEE_LEFT (KNIE_LINKS)|HIP_LEFT (HÜFTE_LINKS)|
|20|ANKLE_LEFT (KNÖCHEL_LINKS)|KNEE_LEFT (KNIE_LINKS)|
|21|FOOT_LEFT (FUSS_LINKS)|ANKLE_LEFT (KNÖCHEL_LINKS)|
|22|HIP_RIGHT (HÜFTE_RECHTS)|PELVIS (BECKEN)|
|23|KNEE_RIGHT (KNIE_RECHTS)|HIP_RIGHT (HÜFTE_RECHTS)|
|24|ANKLE_RIGHT (KNÖCHEL_RECHTS)|KNEE_RIGHT (KNIE_RECHTS)|
|25|FOOT_RIGHT (FUSS_RECHTS)|ANKLE_RIGHT (KNÖCHEL_RECHTS)|
|26|HEAD|NECK (HALS)|
|27|NOSE (NASE)|HEAD|
|28|EYE_LEFT (AUGE_LINKS)|HEAD|
|29|EAR_LEFT (OHR_LINKS)|HEAD|
|30|EYE_RIGHT (AUGE_RECHTS)|HEAD|
|31|EAR_RIGHT (OHR_RECHTS)|HEAD|

Daten sind von [Microsoft](https://learn.microsoft.com/de-de/azure/kinect-dk/body-joints)

### PyKinect-Version

Die Implementierung dieser Version basiert auf https://github.com/ibaiGorordo/pyKinectAzure. Für evtl. Kinect-Fehler, bei diesem Repo in die Issues schauen.

#### Bekannte Bugs:

- Wenn keine Person im Frame ist, wirft das Skeleton-Model der Kinect einen Fehler der den Logger nicht beendet, aber .
- Wenn der PC neu gestartet oder etwas Umgesteckt wird, müssen die IDs der einzelnen Kameras überprüft werden. Es muss sichergestellt werden, dass die ID, die die Kinect ansprechen würde nicht als WebcamStream-Klasse instanziiert wird.
- Die Log-Rate ist nicht konstant (gegen Ende der Messung mehr Schwankung) -> multiprocessing kann hier helfen. Ein Beispiel dazu ist im debug Ordner, bin noch nicht dazu gekommen den ganzen Logger so umzuschreiben, aber könnte man noch testen
- Der Logger braucht noch ein Labeling-Interface. Um eines zu schreiben was Python-Kompatibel ist und nicht zu viel Ressourcen braucht, bietet es sich an, ein HTML/CSS Interface zu implementieren (bzw. ChatGPT implementieren lassen anhand von Abbildung), dass über JS-Logik Python functions lokal aufruft und so gelabelt wird (vom Prinzip her z.B. so wie hier https://dev.to/kodark/creating-a-modern-gui-for-your-python-application-2mlp; gibt aber möglicherweise auch Packages dafür, die das vereinfachen). Andere Möglichkeiten sind Pythons eigene GUI-Library Tkinter oder PyQT (aufwändiger).
- (Optional) Die GUI sollte am Anfang auch dafür genutzt werden können Participant IDs und weitere Infos einzugeben und dann den Logger zu starten.
- Tiefenbilder Pixelwerte checken


#### UI
Im Ordner der Python Version befindet sich ein Unterordner UI. In diesem befinden sich:  

UI  
|_Images  
&nbsp; &nbsp; &nbsp;   |_diverse Hintergründe  
|_Function.js  
|_Labeling.html  
|_StartScreen.html  
|_Style.css  

Zu öffenen ist StartScreen.html.

#### Starten des Loggers
* Zuerst sicherstellen, dass alle Logitech Cams im LogiTune Tool (falls nicht installiert, runterladen) zu sehen sind
* Falls es weniger als 5 sind, alle USB Kabel neu einstecken aber Verteilung der Kabel auf den USB Ports belassen wie im Bild zu sehen
* Es sollten die Camera Views in Logitech geöffnet und Lichtverhältnisse nach eigenen Wünschen angepasst werden, da AutoExposure und Low Light Adaptation für 30Hz Aufnahmen aus sein muss (kann z.B. erreicht werden in dem man Auto Exposure und Low Light Adaptation kurz an und dann wieder ausschaltet oder manuell über die Slider)
* Wenn alle Kameras da sind, cam_setup.py ausführen. Sollte hier eine Kamera fehlen --> nochmal alles neu einstöpseln oder schauen, ob evtl. die Kinect nicht an einem USB 3.0 Port hängt
* Wenn cam_setup.py 6 Kameras (5 Logitech + 1 Kinect) findet, kann die Aufnahme gestartet werden
* Dafür multi_processing_main.py ausführen, Participant ID eingeben, Labeling Interface starten und dort Participant ID eingeben (Beide Teile funktionieren unabhängig voneinander, heißt wenn das Interface geschlossen wird muss der Logger trotzdem in der Konsole über 'Enter' gestoppt werden) 
