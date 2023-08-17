import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

imageOne = cv.imread("/Users/PedroVitorPereira/Documents/GitHub/Masters-in-Computer-Science/Implementation-of-Homography-and-RASAC-algorithms-to-obtain-panoramic-images/images/image_1.1.png")
imageTow  =  cv.imread("/Users/PedroVitorPereira/Documents/GitHub/Masters-in-Computer-Science/Implementation-of-Homography-and-RASAC-algorithms-to-obtain-panoramic-images/images/image_1.2.png")

#One
result = cv.addWeighted(imageOne, 0.3, imageTow, 0.7, 0.0)

cv.imshow('blend', result)
cv.waitKey(0)
#cv.imshow('img1', imageOne)
#cv.imshow('img2', imageTow)


#Tow

print(f"7 - Blending Images using Seamless Cloning")

# Convert images to the format supported by cv.seamlessClone()
img1 = cv.cvtColor(imageOne, cv.COLOR_BGR2GRAY).astype(np.uint32)
img2 = cv.cvtColor(imageTow, cv.COLOR_BGR2GRAY).astype(np.uint32)

# Get the image shapes
img_shape = img1.shape

# Create a mask for the blending region
mask = np.ones(img_shape, dtype=np.uint8)

# Get the center of the left image
center = (img_shape[1] // 2, img_shape[0] // 2)

# Perform seamless cloning to blend the images
result = cv.seamlessClone(img1, img2, mask, center, cv.NORMAL_CLONE)

plt.imshow(result)
plt.title("Blending image using Seamless Cloning")
plt.show()
