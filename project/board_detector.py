import cv2
import numpy as np

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

    contours.sort(key=cv2.contourArea)
    top3 = []
    for i, cnt in enumerate(contours[-3:]):
        convex = cv2.convexHull(cnt)
        approx = cv2.approxPolyDP(convex, epsilon(convex), True)
        approx.shape = (approx.shape[0], approx.shape[2])
        # get rid of polygons touching bottom
        if np.any(frame.shape[0] - approx.T[1] < 200):
            continue
        # TODO: heuristic for getting rid of closet detection.
        # can use aspect ratio, size, fact that it is far from edge
        top3.append(approx)
    for approx in top3:
        cv2.drawContours(frame, [approx], 0, (255, 255, 255), 9)
    if debug:
        cv2.imshow("Source", gray)
        cv2.imshow("canny", canny_edges)
        cv2.imshow("dilated", dilated_edges)
        cv2.imshow("negative", negated)
        cv2.imshow("after contours", frame)
        while True:
            k = cv2.waitKey() & 0xff
            if k == ord('q'):
                break

    return top3, frame


if __name__ == "__main__":
    im = cv2.imread("../snaps/11.png")
    findBoard(im, True)
