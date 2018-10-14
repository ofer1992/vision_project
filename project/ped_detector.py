import cv2
import numpy as np

cap = cv2.VideoCapture('../les2.mp4')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


while 1:
    ret, frame = cap.read()
    if not ret:
        break
    factor = 4.
    small = cv2.resize(frame, (0,0), fx=1/factor, fy=1/factor)

    # rects, weights = hog.detectMultiScale(small, winStride=(4, 4),
    #     padding=(8, 8), scale=4.)
    rects, weights = hog.detectMultiScale(small)
    rects = [[int(i*factor) for i in r] for r in rects]
    for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('vid',frame)
    k = cv2.waitKey(20) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()