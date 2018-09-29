import numpy as np
import cv2
from trackers import TemplateTracker


class BoardTracker:

    board_id = 0
    RECTIFIED_CORNERS = {'tl':(0,0),
                         'tr':(0,300),
                         'bl':(750,0),
                         'br':(750,300)}

    def __init__(self, frame, tl, tr, bl, br):
        self.board_id = BoardTracker.board_id
        BoardTracker.board_id += 1
        self.corner_trackers = {
            'tl': TemplateTracker(frame, tl),
            'tr': TemplateTracker(frame, tr),
            'bl': TemplateTracker(frame, bl),
            'br': TemplateTracker(frame, br)
        }
        self.bg = None
        self.counter = 0
        self.moving = False

    def apply_motion(self, dx, dy):
        """
        move all corners by specified motion.

        this is the calculated motion of the background, under the
        assumption of uniform motion under rotation (probably not true)
        """
        for t in self.corner_trackers:
            self.corner_trackers[t].apply_motion(dx, dy)

    def apply_template_tracking(self, frame):
        """
        improve location estimation of corners using template tracking.
        :param frame:
        :return:
        """
        for t in self.corner_trackers.values():
            if 0 < t.x < frame.shape[1] and 0 < t.y < frame.shape[0]:
                t.track_new_loc(frame)
                t.draw_rect(frame)
            # self.corner_trackers[t].track_new_loc(frame)

    def rectify(self, frame, moving):
        x0 = frame.shape[1]
        y0 = frame.shape[0]
        big = np.zeros(tuple([i * 3 for i in frame.shape[:2]]+[frame.shape[2]]), dtype='uint8')
        big[y0:y0+frame.shape[0],x0:x0+frame.shape[1]] = frame
        # pts1 = np.float32([[x0+t.x, y0+t.y] for t in self.corner_trackers.values()])
        pts1 = np.float32([[x0+self.corner_trackers[c].x, y0+self.corner_trackers[c].y] for c in self.corner_trackers])
        pts2 = np.float32([self.RECTIFIED_CORNERS[c] for c in self.corner_trackers])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(big,M,(750,300))
        if self.bg is None:
            self.bg = dst
        elif self.counter % 25 == 0 and not moving:
            self.bg[dst!=0] = dst[dst!=0]
        dst[dst==0] = self.bg[dst==0]
        self.counter+=1

        cv2.imshow("board #"+str(self.board_id), dst)
        # out.write(dst)

