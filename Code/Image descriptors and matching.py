import cv2 as cv
import  numpy as np
import matplotlib.pyplot as plt

def LoadIamge(path):

    image = cv.imread(path)
    return image

def resingImage(image):

    image  = cv.resize(image,(500,500), cv.INTER_LINEAR)
    return image

def BGR2GRAY(image: object) -> object:

    grayScaleimage  = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    return grayScaleimage

def ShowImage(image):

    cv.imshow("Display Image",image)
    cv.waitKey(0)

def SIFTKeypointsDetetor(grayScaleimage):

    sift = cv.SIFT_create()

    #keyPoints = sift.detect(grayScaleimage, None)

    #SIFTImage = cv.drawKeypoints(image, keyPoints, image)

    return sift.detectAndCompute(grayScaleimage,None)

def MachingSIFTKeypoints(ImageOne, ImageTwo):

    correspondingPoints = []

    keyPointsOne, descriptorsOne = SIFTKeypointsDetetor(ImageOne)

    keyPointsTwo, descriptorsTwo = SIFTKeypointsDetetor(ImageTwo)

    bf = cv.BFMatcher_create(cv.NORM_L1,crossCheck=True)

    matches = bf.match(descriptorsOne,descriptorsTwo)

    matches = sorted(matches,key = lambda  x:x.distance)

    matchedImage = cv.drawMatches(ImageOne,keyPointsOne,ImageTwo,keyPointsTwo, matches[:100], ImageOne, flags=2)

    cv.imwrite("images/Result.jpeg", matchedImage)

    return matchedImage



traningImage = LoadIamge("/Users/PedroVitorPereira/Documents/GitHub/Masters-in-Computer-Science/Job one: panoramic photo/images/image_1.2.png")

matchingImage = LoadIamge("/Users/PedroVitorPereira/Documents/GitHub/Masters-in-Computer-Science/Job one: panoramic photo/images/image_1.1.png")

grayTraningimage= BGR2GRAY(traningImage)

graymachingImage = BGR2GRAY(matchingImage)

matchedImage = MachingSIFTKeypoints(traningImage, matchingImage)

ShowImage(matchedImage)






