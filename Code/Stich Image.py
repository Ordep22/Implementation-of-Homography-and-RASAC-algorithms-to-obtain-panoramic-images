import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random


# Defines


class ImageProcess:
    def __init__(self):
        self.src_pts = None
        self.dst_pts = None
        self.image = None
        self.Paths  = None

    def LoadImage(self, Paths):
        self.Paths = Paths

        print("1- Loading images:")

        self.image = []

        for x in Paths:
            self.image.append(cv.imread(x))

        plt.subplot(1, 2, 1)
        plt.imshow(self.image[0], 'gray')
        plt.title("Image One")

        plt.subplot(1, 2, 2)
        plt.imshow(self.image[1], 'gray')
        plt.title("Image Tow")

        # Show loaded images
        plt.show()

        self.FindDescriptosKeipoints()

    def ImageShape(self):

        imageOne = ImageProcess.image[0]

        imageTow = ImageProcess.image[1]

        (hegihtOne, widthOne) = imageOne.shape[:2]

        (hegihtTow, widthTow) = imageTow.shape[:2]

        return (hegihtOne, widthOne), (hegihtTow, widthTow)

    def FindDescriptosKeipoints(self):

        print("2 - Find descriptors and keypoits")

        # Initializing descriptors
        sift = cv.SIFT_create()

        # Find the keypoints and descriptors with SIFT
        self.kpOne, self.desOne = sift.detectAndCompute(self.image[0], None)
        self.kpTow, self.desTow = sift.detectAndCompute(self.image[1], None)

        # Find good Matches from key points and descripstors
        self.FindGoodMatches()

    def FindGoodMatches(self):

        print("3 - Find the good matches")

        self.src_pts = [] #point's image one

        self.dst_pts = []#points image tow

        self.goodMatches = []

        self.goodMatchespostions = []

        self.Homography = Homography()

        indexParamentrs = dict(algorithm=1, trees=5)

        searchParams = dict(check=50)

        flann = cv.FlannBasedMatcher(indexParamentrs, searchParams)

        # Calculating the matches points from drecriptors
        matches = flann.knnMatch(self.desOne, self.desTow, k=2)

        # Find the good matches
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                self.goodMatches.append(m)

         # Show the number of corresponces implemented
        print(f"The number of correspondeces implementade is: {len(self.goodMatches)}")

        self.src_pts = (([self.kpOne[m.queryIdx].pt for m in self.goodMatches]))

        self.dst_pts = (([self.kpTow[m.trainIdx].pt for m in self.goodMatches]))

        for x in range(0, len(self.src_pts), 1):
            src = list(self.src_pts[x])
            src = [int(src[i]) for i in range(0, len(src), 1)]

            dts = list(self.dst_pts[x])
            dts = [int(dts[i]) for i in range(0, len(dts), 1)]

            self.goodMatchespostions.append([src, dts])

          

        # Draw matches points in a image
        self.DrawMatchesPoints()

        # Find Homography by RANSAC algothim
        self.Homography.Homography(self.goodMatchespostions)

    def DrawMatchesPoints(self, outImage=None):

        print("4 - Drawing the matches points")

        matchedImage = cv.drawMatches(self.image[0], self.kpOne, self.image[1], self.kpTow, self.goodMatches[0:100],
                                      outImage, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.imshow(matchedImage)
        plt.title("Image metches")
        plt.show()

class Homography:

    def __init__(self):
        self.goodMatchespostions = None
        self.goodMatches = None

    def Homography(self, goodMatchespostions):

            global run
            print("5 - Computing the best homograpy by RANSAC")

            self.PosProssesingImage = PosProssesingImage()

            # RANSAC constants
            threshold = 5.0
            maxInlier = 0
            bestHomography = None
            numRandomsubSample = 4
            errorAccumulated = 0

            # Variables of method Homography
            ImageOnepoints = []
            ImageTowpoints = []

            for dstPoint, srcPoint in goodMatchespostions:
                ImageOnepoints.append(dstPoint)
                ImageTowpoints.append(srcPoint)

            ImageOnepoints = np.array(ImageOnepoints)
            ImageTowpoints = np.array(ImageTowpoints)

            numSample = len(goodMatchespostions)
            minInliers = int(numSample * 0.6)  # Minimum inliers for adaptation
            maxIterations = 10000  # Maximum iterations

            # RANSAC algorithm, find the best fit homography
            for run in range(maxIterations):

                if run > 0 and numInlier >= minInliers:
                    break  # Stop early if enough inliers found

                SubSampleIdx = random.sample(range(numSample), numRandomsubSample)

                homography = self.SolveHomography(ImageTowpoints[SubSampleIdx], ImageOnepoints[SubSampleIdx])

                numInlier = 0

                # Find the best Homography with the maximum number of inliers
                for i in range(numSample):

                    if i not in SubSampleIdx:

                        concateCoor = np.hstack((ImageTowpoints[i], [1]))  # Add z-axis as 1

                        dstCoor = homography @ concateCoor.T  # Calculate the coordination after transforming to destination image

                        if dstCoor[2] <= 1e-8:
                            continue

                        dstCoor = dstCoor / dstCoor[2]

                        errorAccumulated += np.linalg.norm(dstCoor[:2] - ImageOnepoints[i])

                        if np.linalg.norm(dstCoor[:2] - ImageOnepoints[i]) < threshold:
                            numInlier += 1

                if numInlier > maxInlier:
                    maxInlier = numInlier
                    bestHomography = homography

            percentual_inliers = (numInlier / numSample) * 100

            percentual_outliers = ((numSample - numInlier) / numSample) * 100

            print(f"The number of Maximum Inliers: {maxInlier}")

            print(f"Number of iterations: {run + 1}")

            print(f"Percentage inliers {percentual_inliers}")

            print(f"Average cumulative error {errorAccumulated/numInlier}")

            # Creating the panoramic image
            self.PosProssesingImage.Warp(bestHomography)


    def SolveHomography(self, originalPlane, newPlane):

        global homographyMatrices
        try:
            A = []
            for r in range(len(originalPlane)):
                A.append([-originalPlane[r, 0], -originalPlane[r, 1], -1, 0, 0, 0, originalPlane[r, 0] * newPlane[r, 0],
                          originalPlane[r, 1] * newPlane[r, 0], newPlane[r, 0]])
                A.append([0, 0, 0, -originalPlane[r, 0], -originalPlane[r, 1], -1, originalPlane[r, 0] * newPlane[r, 1],
                          originalPlane[r, 1] * newPlane[r, 1], newPlane[r, 1]])

            # Solve s ystem of linear equations Ah = 0 using SVD
            u, s, vt = np.linalg.svd(A)

            # Pick H from last line of vt
            homographyMatrices = np.reshape(vt[8], (3, 3))

            # Normalization, let H[2,2] equals to 1
            homographyMatrices = (1 / homographyMatrices.item(8)) * homographyMatrices
        except:
            print("Error occur!")

        return homographyMatrices


# noinspection PyUnreachableCode
class PosProssesingImage:
    def __init__(self):
        pass

    def Warp(self, homography_matrix):

        print(f"6 - Warping images")

        self.PosProssesingImage = PosProssesingImage()

        # Geting the image´s height and width
        (hegihtOne, widthOne), (hegihtTow, widthTow) = ImageProcess.ImageShape()

        # create the big image accroding the image´s height and width
        # create the (stitch)big image accroding the imgs height and width

       #Here ti's necessary to create a shape that was the same os a original imagens

        stitch_img = np.zeros((max(hegihtOne, hegihtTow), widthOne, 3), dtype="int")

        # Transform Right image(the coordination of right image) to destination
        # iamge(the coordination of left image) with homography_matrix
        inv_H = np.linalg.inv(homography_matrix)

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

        plt.imshow(stitch_img)
        plt.title("Stitch Image")
        plt.show()

        # create the Blender object to blending the image
        self.Blending([ImageProcess.image[0], stitch_img])


    def Blending(self, imgs):

        print(f"7 - Blending Images")

        blendingImage = cv.addWeighted(imgs[0], 0.3, imgs[1], 0.7, 0.0,dtype=cv.CV_8U)

        plt.imshow(blendingImage)

        plt.title("Blending image")

        plt.show()


if __name__ == "__main__":
    files = [
        r"/Users/PedroVitorPereira/Documents/GitHub/Masters-in-Computer-Science/Implementation-of-Homography-and-RASAC-algorithms-to-obtain-panoramic-images/images/image_5.2.png",
        r"/Users/PedroVitorPereira/Documents/GitHub/Masters-in-Computer-Science/Implementation-of-Homography-and-RASAC-algorithms-to-obtain-panoramic-images/images/image_5.1.png",
        ]

    ImageProcess = ImageProcess()

    ImageProcess.LoadImage(files)

    print("Finish algorithm")
