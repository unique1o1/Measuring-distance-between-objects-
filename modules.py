
import cv2

import numpy as np


def sort_contours(cnts: list, *, reverse: bool=False):
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda x: x[1][0], reverse=False))
    return cnts, boundingBoxes


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def box_point(_):
    a = []
    a.append([_[0], _[1]])
    a.append([_[0]+_[2], _[1]])
    a.append([_[0]+_[2], _[1]+_[3]])
    a.append([_[0], _[1]+_[3]])

    return np.asarray(a)


def order_point(x):
    xSorted = x[np.argsort(x[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    (tl, bl) = leftMost[np.argsort(leftMost[:, 1]), :]
    (tr, br) = rightMost[np.argsort(rightMost[:, 1]), :]
    return np.array([tl, tr, br, bl], dtype="float32")


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized
