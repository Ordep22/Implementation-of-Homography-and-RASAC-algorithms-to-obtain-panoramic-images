import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Defines
MIN_MATCH_COUNT = 10

FLANN_INDEX_KDTREE = 1

imageOne = cv.imread(
    "/Users/PedroVitorPereira/Documents/GitHub/Masters-in-Computer-Science/Job one: panoramic photo/images/image_1.1.png",
    cv.IMREAD_GRAYSCALE)

imageTow = cv.imread(
    "/Users/PedroVitorPereira/Documents/GitHub/Masters-in-Computer-Science/Job one: panoramic photo/images/image_1.2.png",
    cv.IMREAD_GRAYSCALE)

# Intiate SIFT detectors
sift = cv.SIFT_create()

# Finnd the keypoints and decriptors with SIFT

kpOne, desOne = sift.detectAndCompute(imageOne, None)

kpTow, desTow = sift.detectAndCompute(imageTow, None)

indexParamentrs = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(check=50)

flann = cv.FlannBasedMatcher(indexParamentrs, searchParams)

matches = flann.knnMatch(desOne, desTow, k=2)

# Store all the good matches as per Lowe's ration test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:

    src_pts = np.float32([kpOne[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpTow[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)


    # From this point it's no usualy, becou it's going to implemented. But for test it's ok

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = imageOne.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    img2 = cv.polylines(imageTow, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

img3 = cv.drawMatches(imageOne, kpOne, imageTow, kpTow, good, None, **draw_params)
plt.imshow(img3, 'gray'), plt.show()
