
import numpy as np
import argparse
import cv2


def sort_contours(cnts, *, reverse=False):
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda x: x[1][0], reverse=False))
    return cnts, boundingBoxes


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
                help="width of the left-most object in the image (in inches)")
args = ap.parse_args()
image = cv2.imread(args.image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours in the edge map
cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)[1]
cnts, boundingBoxes = sort_contours(cnts, reverse=False)
colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
          (255, 0, 255))
refobj = None


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


for (c, boundingBoxes) in zip(cnts, boundingBoxes):
    if cv2.contourArea(c) < 100:
        continue

    box = box_point(boundingBoxes)
    box = order_point(box)
    print(box)
    cx = np.average(box[:, 0])
    cy = np.average(box[:, 1])
