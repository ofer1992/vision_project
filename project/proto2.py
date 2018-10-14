import cv2
import numpy as np

from utils import imshow, build_board
from trackers import Tracker, FeaturePointManager
from legacy import MotionHistogram
# from boardtracker import BoardTracker

# cap.get(cv2.CAP_PROP_POS_FRAMES)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture("../les1.mp4")

# fast forward to frame with full board in frame
for _ in range(100):
    ret = cap.grab()
# Take first frame and find corners in it
ret, old_frame = cap.read()
# old_frame = cv2.resize(old_frame, (0,0), fx=0.5,fy=0.5)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

motion_tracker = Tracker(old_frame)
motion_hist = MotionHistogram()
boards = []
mask = np.zeros_like(old_frame)

while(1):
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # optical flow
    print len(FeaturePointManager._points)
    """
    p0 = FeaturePointManager.get_visible_points()
    if p0.shape[0] < 3:
        FeaturePointManager.generate_points_for_frame(old_frame)
        p0 = FeaturePointManager.get_visible_points()
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    mode_x, mode_y = motion_hist.mode(good_new, good_old)
    for p in FeaturePointManager._points:
        p.x += mode_x
        p.y += mode_y
    """
    # motion_hist.
    h = motion_tracker.find_homography(frame)
    if h is not None:
        FeaturePointManager.apply_homography(h, frame_gray, only_non_visible_points=False)
    # for board in boards:
    #     board.apply_motion(dx, dy)
    #     board.apply_template_tracking(frame_gray)
    #     board.rectify(frame, camera_moving)
    # mask = np.uint8(0.9 * mask)
    # mask = np.zeros_like(frame)
    displayed = frame.copy()
    FeaturePointManager.draw_points_on_frame(displayed)
    # motion_tracker.draw_points_on_frame(mask)
    # for board in boards:
    #     board.draw_corners(frame)

    cv2.imshow('frame', displayed)
    k = cv2.waitKey(1) & 0xff
    if k == ord('b'):
        tmp = build_board(frame)
        if tmp is not None:
            boards.append(BoardTracker(frame_gray, tl=tmp[0], bl=tmp[1], tr=tmp[2],
                                       br=tmp[3]))
        break

    # Now update the previous frame and previous points
    old_frame = frame.copy()
    old_gray = frame_gray.copy()

cv2.destroyAllWindows()
cap.release()
# out.release()
