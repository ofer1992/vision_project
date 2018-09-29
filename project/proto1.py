import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import imshow
from trackers import TemplateTracker, CornerTracker, OpticalFlowTracker
from boardtracker import  BoardTracker


    


    
cap = cv2.VideoCapture("../lecture.mp4")

# fast forward to frame with full board in frame
for _ in range(120):
    a,b = cap.read()




# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# mask allows choice of feature points only from upper part of frame
corner_mask = np.zeros(old_frame.shape[:2], dtype='uint8')
corner_mask[:150,:] = 255
# imshow("mask",corner_mask, gray=True)

motion_tracker = OpticalFlowTracker(old_gray, corner_mask)
# ct = CornerTracker(cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY))

# initial corner locations.
corners = np.array([[285,70],[1170,50],[300,420],[1160,400]], dtype='float64')
board1 = BoardTracker(old_gray, tl=corners[0], bl=corners[1], tr=corners[2],
                      br=corners[3])
# setting tracker templates
# corner_templates2 = []
# for c in np.uint16(corners):
#     corner_templates2.append(TemplateTracker(old_frame, c[0], c[1]))

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (750,300))

# bg = None
# counter = 0
# moving = False
# def rectify(frame):
#     global bg
#     global counter
#     global moving
#     x0 = frame.shape[1]
#     y0 = frame.shape[0]
#     big = np.zeros(tuple([i * 3 for i in frame.shape[:2]]+[frame.shape[2]]), dtype='uint8')
#     big[y0:y0+frame.shape[0],x0:x0+frame.shape[1]] = frame
#     pts1 = np.float32([[x0+t.x, y0+t.y] for t in corner_templates2])
#     pts2 = np.float32([[0,0],[750,0],[0,300],[750,300]])
#     M = cv2.getPerspectiveTransform(pts1,pts2)
#     dst = cv2.warpPerspective(big,M,(750,300))
#     if bg is None:
#         bg = dst
#     elif counter % 25 == 0 and not moving:
#         bg[dst!=0] = dst[dst!=0]
#     dst[dst==0] = bg[dst==0]
#     counter+=1
#
#     cv2.imshow("big", dst)
#     # out.write(dst)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)


while(1):
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
        
    mode_x, mode_y = motion_tracker.calc_flow(frame_gray)
    # mode_x, mode_y = ct.traGck(frame_gray)
    moving = abs(mode_x) > 3

    board1.apply_motion(mode_x, mode_y)
    board1.apply_template_tracking(frame_gray)
    board1.rectify(frame, moving)
    # for t in corner_templates2:
    #     t.apply_motion(mode_x, mode_y)
    #     if 0 < t.x < frame.shape[1] and 0 < t.y < frame.shape[0]:
    #         t.track_new_loc(frame)
    #         t.draw_rect(frame)
        
    # rectify(frame)

    motion_tracker.draw_points_on_frame(frame)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()

cv2.destroyAllWindows()
cap.release()
# out.release()
