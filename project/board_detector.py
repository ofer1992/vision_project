import cv2
import numpy as np
from utils import compareBoards

LEFT_COLOR = (255,0,0)
RIGHT_COLOR = (0,0,255)
UNKNOWN_COLOR = (150,50,101)

def epsilon(cnt):
    return 0.01*cv2.arcLength(cnt, True)

def findBoard(frame, debug=False):
    frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) != 2 else frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_edges = cv2.Canny(blurred, 100, 250, None, 3)
    kernel = np.ones((5,5),np.uint8)
    dilated_edges = cv2.dilate(canny_edges, kernel, iterations=1)
    negated = 255 - dilated_edges
    f, contours,h = cv2.findContours(negated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours.sort(key=cv2.contourArea, reverse=True)
    top3 = []
    for i, cnt in enumerate(contours[:3]):
        convex = cv2.convexHull(cnt)
        approx = cv2.approxPolyDP(convex, epsilon(convex), True)
        approx.shape = (approx.shape[0], approx.shape[2])
        # get rid of polygons touching bottom
        if np.any(frame.shape[0] - approx.T[1] < 200):
            continue
        # get rid of closet
        points_on_edge = approx[approx[:,0] == frame.shape[1]-1]
        if len(points_on_edge) == 2:
            if abs(points_on_edge[0,1] - points_on_edge[1,1]) < 350:
                print "threw", approx, "because of closet heuristic"
                continue
        # can use aspect ratio, size, fact that it is far from edge
        top3.append(approx)
    colors = [UNKNOWN_COLOR] * len(top3)
    if len(top3) == 1:
      print top3
      if np.any(top3[0][:,0] == 0):
          colors = [RIGHT_COLOR]
      elif np.any(top3[0].T[0] == frame.shape[0]-1):
          colors = [LEFT_COLOR]
    if len(top3) == 2:
        if top3[0][0,0] < top3[1][0,0]:
            colors = [LEFT_COLOR, RIGHT_COLOR]
        else:
            colors = [RIGHT_COLOR, LEFT_COLOR]
    for i, approx in enumerate(top3):
        cv2.drawContours(frame, [approx], 0, colors[i], 9)
    if debug:
        cv2.imshow("Source", gray)
        cv2.imshow("canny", canny_edges)
        cv2.imshow("dilated", dilated_edges)
        cv2.imshow("negative", negated)
        cv2.imshow("after contours", frame)
        while True:
            k = cv2.waitKey() & 0xff
            if k == ord('q'):
                cv2.destroyAllWindows()
                break

    return top3, frame, colors

def rectifyBoard(im, contour):
    b_right = cv2.imread('../b1.png', cv2.COLOR_BGR2GRAY)
    b_left = cv2.imread('../b2.png', cv2.COLOR_BGR2GRAY)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    det = [box]
    corners = ['tl', 'bl', 'tr', 'br']
    hight = np.max(det[0].T[1]) - np.min(det[0].T[1])
    width = np.max(det[0].T[0]) - np.min(det[0].T[0])
    # print hight, width
    rectified_corners = {'tl':(0, 0),
                         'bl':(0,hight),
                         'tr':(width,0),
                         'br':(width,hight)}
    coords = det[0].tolist()
    board = {}
    coords.sort(key=lambda x: x[0])
    board['tl'], board['bl'] = sorted(coords[:2], key=lambda x: x[1])
    board['tr'], board['br'] = sorted(coords[2:4], key=lambda x: x[1])
    # print board

    pts1 = np.float32([board[c] for c in corners])
    pts2 = np.float32([rectified_corners[c] for c in corners])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(im,M,(width,hight))
    if  compareBoards(dst, b_left) < compareBoards(dst, b_right):
        print "left"
    else:
        print "right"

if __name__ == "__main__":
    im = cv2.imread("../snaps/21.png")
    def rep_and_rectify():
        corners = ['tl', 'bl', 'tr', 'br']
        det, im_with_contours = findBoard(im, True)
        hight = np.max(det[0].T[1]) - np.min(det[0].T[1])
        width = np.max(det[0].T[0]) - np.min(det[0].T[0])
        print hight, width
        rectified_corners = {'tl':(0, 0),
                             'bl':(0,hight),
                             'tr':(width,0),
                             'br':(width,hight)}
        coords = det[0].tolist()
        board = {}
        coords.sort(key=lambda x: x[0])
        board['tl'], board['bl'] = sorted(coords[:2], key=lambda x: x[1])
        board['tr'], board['br'] = sorted(coords[2:4], key=lambda x: x[1])
        print board

        pts1 = np.float32([board[c] for c in corners])
        pts2 = np.float32([rectified_corners[c] for c in corners])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(im,M,(width,hight))
        cv2.drawContours(im, [det[0]], 0, (255, 255, 255), -1)
        cv2.imshow('board', im)
        cv2.imshow('rectified', dst)
        cv2.imwrite('right_rep.png', dst)
        while True:
            k = cv2.waitKey() & 0xff
            if k == ord('q'):
                cv2.destroyAllWindows()
                break

    det, _a, _b = findBoard(im, False)
    for d in det:
        rectifyBoard(im, d)
    print det
    cv2.imshow('im', im)
    while True:
        k = cv2.waitKey() & 0xff
        if k == ord('q'):
            cv2.destroyAllWindows()
            break

