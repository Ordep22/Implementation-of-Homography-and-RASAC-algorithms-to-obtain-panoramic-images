import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random

from openai.cli import Image


# Defines


class ImageProcess:
    def __init__(self):
        self.src_pts = None
        self.dst_pts = None
        self.image = None

    def LoadImage(self, Paths):
        self.image = []

        for x in Paths:
            self.image.append(cv.imread(x))


        plt.subplot(1,2,1)
        plt.imshow(self.image[0],'gray')
        plt.title("Image One")

        plt.subplot(1,2,2)
        plt.imshow(self.image[1],'gray')
        plt.title("Image Tow")

        #Show loaded images
        plt.show()


        self.FindDescriptosKeipoints()

    def ImageShape(self):

        imageOne = ImageProcess.image[0]

        imageTow = ImageProcess.image[1]

        (hegihtOne, widthOne) = imageOne.shape[:2]

        (hegihtTow, widthTow) = imageTow.shape[:2]

        return (hegihtOne, widthOne), (hegihtTow, widthTow)

    def FindDescriptosKeipoints(self):

        # Initializing descriptors
        sift = cv.SIFT_create()

        # Find the keypoints and descriptors with SIFT
        self.kpOne, self.desOne = sift.detectAndCompute(self.image[0], None)
        self.kpTow, self.desTow = sift.detectAndCompute(self.image[1], None)

        #Find good Matches from key points and descripstors
        self.FindGoodMatches()

    def FindGoodMatches(self):

        self.src_pts = []

        self.dst_pts = []

        self.goodMatches = []

        self.goodMatchespostions = []

        self.Homography = Homography()

        indexParamentrs = dict(algorithm=1, trees=5)

        searchParams = dict(check=50)

        flann = cv.FlannBasedMatcher(indexParamentrs, searchParams)

        #Calculating the matches points from drecriptors
        matches = flann.knnMatch(self.desOne, self.desTow, k=2)

        # Find the good matches
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                self.goodMatches.append(m)

        self.src_pts = (([self.kpOne[m.queryIdx].pt for m in self.goodMatches]))

        self.dst_pts = (([self.kpTow[m.trainIdx].pt for m in self.goodMatches]))

        for x in range(0, len(self.src_pts), 1):

            src = list(self.src_pts[x])
            src = [int(src[i]) for i in range(0, len(src), 1)]

            dts = list(self.dst_pts[x])
            dts = [int(dts[i]) for i in range(0, len(dts), 1)]

            self.goodMatchespostions.append([src, dts])

        #Draw matches points in a image
        self.DrawMatchesPoints()

       #Find Homography by RANSAC algothim
        self.Homography.Homography(self.goodMatchespostions)

    def DrawMatchesPoints(self, outImage=None):

        matchedImage = cv.drawMatches(self.image[0], self.kpOne, self.image[1], self.kpTow, self.goodMatches,
                                      outImage, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.imshow(matchedImage, 'gray')
        plt.title("Matches points")
        plt.show()


class Homography:

    def __init__(self):
        pass

    def Homography(self, goodMatchespostions):

        self.PosProssesingImage = PosProssesingImage()

        dstPoints = []
        srcPoints = []

        for dstPoint, srcPoint in goodMatchespostions:
            dstPoints.append(dstPoint)
            srcPoints.append(srcPoint)

        dstPoints = np.array(dstPoints)
        srcPoints = np.array(srcPoints)

        # RANSAC algorithm, selecting the best fit homography
        NumSample = len(goodMatchespostions)
        threshold = 5.0
        Iterations = 4000
        NumRandomSubSample = 4
        MaxInlier = 0
        Best_H = None

        for run in range(Iterations):

            SubSampleIdx = random.sample(range(NumSample), NumRandomSubSample)  # Get the index of random sampling

            homography = self.SolveHomography(srcPoints[SubSampleIdx], dstPoints[SubSampleIdx])

            # Find the best Homography with the maximum number of inliers
            NumInlier = 0

            for i in range(NumSample):

                if i not in SubSampleIdx:

                    concateCoor = np.hstack((srcPoints[i], [1]))  # Add z-axis as 1

                    dstCoor = homography @ concateCoor.T  # Calculate the coordination after transforming to destination image

                    if dstCoor[2] <= 1e-8:  # Avoid division by zero or causing overflow due to very small number
                        continue

                    dstCoor = dstCoor / dstCoor[2]

                    if np.linalg.norm(dstCoor[:2] - dstPoints[i]) < threshold:
                        NumInlier += 1

            if MaxInlier < NumInlier:

                MaxInlier = NumInlier

                Best_H = homography

        print("The Number of Maximum Inliers:", MaxInlier)

        print("The Number of Maximum Outliers:", MaxInlier)

        print("The Number of Maximum Inliers/Allmatche:", MaxInlier)

        #Creating the panoramic image
        self.PosProssesingImage.Warp(Best_H)

    def SolveHomography(self, originalPlane, newPlane):

        try:
            A = []
            for r in range(len(originalPlane)):
                A.append([-originalPlane[r, 0], -originalPlane[r, 1], -1, 0, 0, 0, originalPlane[r, 0] * newPlane[r, 0], originalPlane[r, 1] * newPlane[r, 0], newPlane[r, 0]])
                A.append([0, 0, 0, -originalPlane[r, 0], -originalPlane[r, 1], -1, originalPlane[r, 0] * newPlane[r, 1], originalPlane[r, 1] * newPlane[r, 1], newPlane[r, 1]])

            # Solve s ystem of linear equations Ah = 0 using SVD
            u, s, vt = np.linalg.svd(A)

            # Pick H from last line of vt
            homographyMat = np.reshape(vt[8], (3, 3))

            # Normalization, let H[2,2] equals to 1
            homographyMat = (1 / homographyMat.item(8)) * homographyMat
        except:
            print("Error occur!")

        return homographyMat


# noinspection PyUnreachableCode
class PosProssesingImage:
    def __init__(self):
        pass
    def Warp(self,HomoMat):

        self.PosProssesingImage = PosProssesingImage()

        #Geting the image´s height and width
        (hegihtOne, widthOne),(hegihtTow, widthTow) = ImageProcess.ImageShape()

        # create the big image accroding the image´s height and width
        stitch_img = np.zeros((max(hegihtOne, hegihtTow), widthOne + widthTow, 3),dtype="int")  # create the (stitch)big image accroding the imgs height and width

        # Transform Right image(the coordination of right image) to destination iamge(the coordination of left image) with HomoMat
        inv_H = np.linalg.inv(HomoMat)

        for i in range(stitch_img.shape[0]):
            for j in range(stitch_img.shape[1]):
                coor = np.array([j, i, 1])
                img_right_coor = inv_H @ coor  # the coordination of right image
                img_right_coor /= img_right_coor[2]

                # you can try like nearest neighbors or interpolation
                y, x = int(round(img_right_coor[0])), int(round(img_right_coor[1]))  # y for width, x for height

                # if the computed coordination not in the (hegiht, width) of right image, it's not need to be process
                if (x < 0 or x >= hegihtTow or y < 0 or y >= widthTow):
                    continue
                # else we need the tranform for this pixel
                stitch_img[i, j] = ImageProcess.image[1][x, y]


        # create the Blender object to blending the image
        stitch_img = self.Blending([ImageProcess.image[0], stitch_img])



        plt.imshow(stitch_img.astype(int))
        plt.title("Stitch Image")
        plt.show()


    def Blending(self, imgs):

        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")

        # find the left image and right image mask region(Those not zero pixels)
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1

        # find the overlap mask(overlap region of two image)
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1

        # Plot the overlap mask
        plt.title("overlap_mask")
        plt.imshow(overlap_mask.astype(int), cmap="gray")

        # compute the alpha mask to linear blending the overlap region
        alpha_mask = np.zeros((hr, wr))  # alpha value depend on left image
        for i in range(hr):
            minIdx = maxIdx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j

            if (minIdx == maxIdx):  # represent this row's pixels are all zero, or only one pixel not zero
                continue

            decrease_step = 1 / (maxIdx - minIdx)
            for j in range(minIdx, maxIdx + 1):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))

        linearBlending_img = np.copy(img_right)
        linearBlending_img[:hl, :wl] = np.copy(img_left)
        # linear blending
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(overlap_mask[i, j]) > 0):
                    linearBlending_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[
                        i, j]

        return linearBlending_img



if __name__ == "__main__":
    files = [
        r"C:\Users\pedro.pereira\OneDrive - LUPA\Documentos\25_GitHub\Implementation-of-Homography-and-RASAC-algorithms-to-obtain-panoramic-images\images\image_1.1.png",
        r"C:\Users\pedro.pereira\OneDrive - LUPA\Documentos\25_GitHub\Implementation-of-Homography-and-RASAC-algorithms-to-obtain-panoramic-images\images\image_1.2.png"]

    ImageProcess = ImageProcess()

    ImageProcess.LoadImage(files)


