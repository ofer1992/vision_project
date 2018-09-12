from matplotlib import pyplot as plt
import cv2


def imshow(title, img, size=8, gray=False):
    'display opencv image'

    fig = plt.figure(figsize=(size, size))
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
