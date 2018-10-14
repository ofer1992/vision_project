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

# im = np.ones((50,50,3), dtype='uint8') * 255
# build_board(im)
