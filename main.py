
import numpy as np
import argparse
import cv2

from scipy.spatial import distance as dist


def sort_contours(cnts, *, reverse=False):
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda x: x[1][0], reverse=False))
    return cnts, boundingBoxes


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


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
    ''' order the points in the contour such that they appear
     in top-left, top-right, bottom-right, and bottom-left
    order'''
    box = order_point(box)
    print(box)
    cx = np.average(box[:, 0])
    cy = np.average(box[:, 1])
    if refObj is None:

        (tl, tr, br, bl) = box
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        refobj = [box, [cx, cy], D / args["width"]]
        continue
    orig = image.copy()
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
    cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

    # stack the reference coordinates and the object coordinates
    # to include the object center
    refCoords = np.vstack([refObj[0], refObj[1]])
    objCoords = np.vstack([box, (cX, cY)])
