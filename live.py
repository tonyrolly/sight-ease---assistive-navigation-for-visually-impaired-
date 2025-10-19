import cv2
import numpy as np
import time
from gtts import gTTS 
# Load Yolo
yolo_net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
image_classes = []
with open("coco.names", "r") as f:
    image_classes = [line.strip() for line in f.readlines()]
yolo_lnames = yolo_net.getLayerNames()
yolo_out_layes = [yolo_lnames[i-1] for i in yolo_net.getUnconnectedOutLayers()]
random_color = np.random.uniform(0, 255, size=(len(image_classes), 3))

# Loading video
vid_cam = cv2.VideoCapture("Blind Man Walking.mp4")

#font dis
font_val = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

import gtts  
import os
from playsound import playsound  



def detect_obj(frame,fcnt):
 

    im_hght, im_wdth, channels = frame.shape

    image_blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    yolo_net.setInput(image_blob)
    det_outs = yolo_net.forward(yolo_out_layes)

    cls_idlist = []
    problist = []
    boxlist = []
    for out in det_outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                #center coordinates 
                center_x = int(detection[0] * im_wdth)
                center_y = int(detection[1] * im_hght)
                w = int(detection[3] * im_wdth)
                h = int(detection[3] * im_hght)

                # Rectangle coordinates
                x = int(center_x - w / 1.8)
                y = int(center_y - h / 1.8)

                boxlist.append([x, y, w, h])
                problist.append(float(confidence))
                cls_idlist.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxlist, problist, 0.4, 0.3)

    for i in range(len(boxlist)):
        if i in indexes:
            x, y, w, h = boxlist[i]
            label = str(image_classes[cls_idlist[i]])
            
            confidence = problist[i]
            color = random_color[cls_idlist[i]]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font_val, 2, color, 2)
            # if(fcnt%100==0 and fcnt!=0):
            #     detlabel=label+" Detected infront of you"
            #     t1 = gtts.gTTS(detlabel)  

            #     t1.save("audio1.mp3")   

            #     playsound("audio1.mp3")  

            #     try:
            #         os.remove("audio1.mp3")
            #     except:
            #         pass


    return frame




while True:
    _, frame = vid_cam.read()
    frame=cv2.resize(frame,(1200,800))

    res=detect_obj(frame,frame_id)
    frame_id += 1
    
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid_cam.release()
cv2.destroyAllWindows()