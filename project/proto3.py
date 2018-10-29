import cv2
import numpy as np
from trackers import DetectorAPI
from utils import alignImages, isRotating
from board_detector import findBoard

RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)



cap = cv2.VideoCapture("../lecture.mp4")

ret, last_frame = cap.read()
last_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

font_kwargs = {
    "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
    "org": (10, 500),
    "fontScale": 1,
    "color": WHITE,
    "thickness": 2,
}

model_path = '/home/tomer/git/vision_project/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
odapi = DetectorAPI(path_to_ckpt=model_path)
bg = np.zeros_like(last_gray)
fgmask = odapi.getHumanMask(last_frame, first_frame_after_cam_movement=True)
bg[fgmask == 0] = last_gray[fgmask == 0]
first_frame_after_cam_movement = False
id = 0

while cap.isOpened():
    for i in range(5): cap.grab()
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rotating = isRotating(gray, last_gray)
    if not rotating:
        if first_frame_after_cam_movement:
            detected, im_contours, colors = findBoard(bg)
            cv2.imshow('contours', im_contours)
            if (bg == 0).sum() < 10000:
                cv2.imwrite("/home/tomer/git/vision_project/snaps/" + str(id) +".png", bg)
                id += 1
                print str(id)+".png written"
            else:
                print "black in bg sum ==", (bg == 0).sum()
            bg = np.zeros_like(gray)
            first_frame_after_cam_movement = False
        fgmask = odapi.getHumanMask(frame, first_frame_after_cam_movement)
        bg[fgmask == 0] = gray[fgmask == 0]
        cv2.imshow('bg', bg)
    else:
        first_frame_after_cam_movement = True
    k = cv2.waitKey(20) & 0xff
    if k == ord('q'):
        cap.release()

    # font_kwargs['color'] = RED if rotating else BLUE
    # cv2.putText(frame, str(diff_sum), **font_kwargs)
    cv2.imshow('vid', frame)

    last_frame = frame
    last_gray = gray

cv2.destroyAllWindows()
