import cv2
import numpy as np
from trackers import DetectorAPI
from utils import alignImages

RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

def diff(curr, last):
    """
    returns binarized diff between two frames and number
    of different pixels.
    """
    THRESH = 100
    diff_img = cv2.absdiff(curr, last)
    _, diff_img = cv2.threshold(diff_img, THRESH, 255, cv2.THRESH_BINARY)
    return diff_img, int(np.sum(diff_img) / 255)


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
only_bg = None
first_iter = True
while cap.isOpened():
    for i in range(5): cap.grab()
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff_im, diff_sum = diff(gray, last_gray)
    font_kwargs['color'] = RED if diff_sum > 10000 else BLUE
    cv2.putText(frame, str(diff_sum), **font_kwargs)
    cv2.imshow('vid', frame)
    if diff_sum <= 20000:
        if first_iter:
            if only_bg is not None:
                warped, h = alignImages(only_bg, gray)
                cv2.imshow('warped', warped)
            else:
                warped = np.zeros_like(gray)
            only_bg = np.zeros_like(gray)
            first_iter = False
        fgmask = odapi.getHumanMask(frame, first_iter)
        only_bg[fgmask == 0] = gray[fgmask == 0]
        cv2.imshow('diff', diff_im)
        cv2.imshow('bg', cv2.addWeighted(only_bg, 0.5, warped, 0.5, 0))
    else:
        first_iter = True
    k = cv2.waitKey(20) & 0xff
    if k == ord('q'):
        cap.release()

    last_frame = frame
    last_gray = gray

cv2.destroyAllWindows()
