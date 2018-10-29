import cv2
import numpy as np
from utils import isRotating, DEFAULT_FONT_KWARGS, waitQ
from board_detector import findBoard
import os
from trackers import DetectorAPI

cap = cv2.VideoCapture("../les1.mp4")
_, last_frame = cap.read()
last_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
first_frame_after_cam_movement = False
id = 0
directory = "/home/tomer/git/vision_project/proto4/"

# while cap.isOpened():
while False:
    for i in range(5): cap.grab()
    ret, frame = cap.read()
    cv2.imshow('vid', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret: break
    rotating = isRotating(gray, last_gray)
    if not rotating:
        if first_frame_after_cam_movement:
            cv2.imwrite(directory + str(id) +".png", frame)
            id += 1
            print id
            first_frame_after_cam_movement = False
    else:
        first_frame_after_cam_movement = True
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'): break

    last_frame = frame
    last_gray = gray

cv2.destroyAllWindows()
model_path = '/home/tomer/git/vision_project/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
odapi = DetectorAPI(path_to_ckpt=model_path)
files = sorted(os.listdir(directory), key=lambda x: int(x[:x.rindex('.')]))[:20]
for file in files:
    frame = cv2.imread(directory+file)
    cv2.imshow(file, frame)
    fgmask = odapi.getHumanMask(frame)
    frame[fgmask != 0] = (0, 0, 0)
    findBoard(frame, True)
    # waitQ()
    cv2.waitKey(100)