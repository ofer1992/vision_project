from numpy.f2py.auxfuncs import throw_error

import cv2
import numpy as np
from utils import imshow

img = cv2.imread("../snaps/2.png", cv2.IMREAD_GRAYSCALE)


# imshow(img)
blurred = cv2.GaussianBlur(img, (5, 5), 0)
thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV )[1]

# imshow(thresh)

blurred = cv2.GaussianBlur(img, (5, 5), 0)
dst = cv2.Canny(blurred, 100, 250, None, 3)
cv2.imshow("canny",dst)

# Copy edges to the images that will display the results in BGR
cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
# cdstP = np.copy(cdst)
cdstP = np.ones_like(dst) * 255

# lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

# if lines is not None:
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#         cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)


linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 100, 100)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,0), 3, cv2.LINE_AA)

cv2.imshow("Source", img)
# cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)


f, contours,h = cv2.findContours(cdstP, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros(img.shape[:2], dtype='uint8')
max_rect = None
max_area = 0
def epsilon(cnt):
    # return 0.08*cv2.arcLength(cnt, True)
    return 0.08*cv2.arecv2.boundingRect(cnt)**.5

def contour_area(cnt):
    # cv2.approxPolyDP(cnt,epsilon(cnt), True)
    return cv2.contourArea(cnt)

# TODO: approx quadrilateral

contours = [cnt for cnt in contours ]#if len(cv2.approxPolyDP(cnt,epsilon(cnt), True)) < 15]
contours.sort(key=contour_area)
for i, cnt in enumerate(contours[-3:]):
    cv2.drawContours(img, [cnt], 0, (255, 255, 255), 3)

# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt, True), True)
    # cv2.drawContours(img, [cnt],0,(255,255,255), -1)
    # print cv2.contourArea(approx)
    # if len(approx)==5:
    #     pass
#             print "pentagon"
#             cv2.drawContours(img,[cnt],0,255,-1)
#         elif len(approx)==3:
#             print "triangle"
#             cv2.drawContours(img,[cnt],0,(0,255,0),-1)
#     elif len(approx)==4:
        # print "square"
    # if cv2.contourArea(cnt) > max_area:
    #     max_rect = cnt
    #     max_area = cv2.contourArea(cnt)
#             cv2.drawContours(img,[cnt],0,(0,0,255),-1)
#         elif len(approx) == 9:
#             print "half-circle"
#             cv2.drawContours(img,[cnt],0,(255,255,0),-1)
#         elif len(approx) > 15:
#             print "circle"
#             cv2.drawContours(img,[cnt],0,(0,255,255),-1)

# cv2.drawContours(img, [max_rect], 0, (255,0,0), -1)
cv2.imshow("after contours", img)
while True:
    k = cv2.waitKey() & 0xff
    if k == ord('q'):
        break
