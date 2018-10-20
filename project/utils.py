from matplotlib import pyplot as plt
import cv2
import numpy as np

imshow_id = 0
def imshow(img, title=None, size=8, gray=False):
    'display opencv image'
    global imshow_id
    if title is None:
        title = str(imshow_id)
        imshow_id += 1
    fig = plt.figure(figsize=(size, size))
    gray = len(img.shape) == 2
    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img[:, :, ::-1])
    plt.title(title + "\n" + str(img.dtype))
    plt.show()


def color_hist(img):
    'plot histogram of colors'
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

def build_board(frame):
    """
    display image and click on corners to specify coordinates.
    expected order is top-left, bottom-left, top-right, bottom-right
    :param frame:
    :return:
    """
    fig = plt.figure(figsize=(20,20))
    plt.imshow(frame[:,:,::-1])
    coords = []
    def on_click(event, coords=coords):
        coords += [(event.xdata, event.ydata)]
        if len(coords) == 4:
            plt.close(fig)
        print coords
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    if len(coords) != 4:
        return None
    return coords

def alignImages(im1, im2):
    """
    warps im1 to im2 and returns homography
    """
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15
    # Convert images to grayscale
    if len(im1.shape) != 2:
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    else:
        im1Gray = im1
    if len(im2.shape) != 2:
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    else:
        im2Gray = im2

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width = im2.shape[0], im2.shape[1]
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

# im = np.ones((50,50,3), dtype='uint8') * 255
# build_board(im)
