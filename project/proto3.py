import cv2
import numpy as np
import matplotlib.pyplot as plt

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

only_bg = None
first_iter = True
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff_im, diff_sum = diff(gray, last_gray)
    font_kwargs['color'] = RED if diff_sum > 10000 else BLUE
    cv2.putText(frame, str(diff_sum), **font_kwargs)
    cv2.imshow('vid', frame)
    if diff_sum <= 10000:
        if first_iter:
            only_bg = np.float32(gray)
            first_iter = False
        cv2.accumulateWeighted(gray, only_bg, 0.05)
        cv2.imshow('diff', diff_im)
    else:
        plt.imshow(cv2.convertScaleAbs(only_bg), cmap='gray')
        plt.show()
        first_iter = True
    k = cv2.waitKey(20) & 0xff
    if k == ord('q'):
        cap.release()

    last_frame = frame
    last_gray = gray

cv2.destroyAllWindows()
