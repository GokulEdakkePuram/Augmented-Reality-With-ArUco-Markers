import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('Room with ArUco Markers-20240324/20221115_113319.jpg')
plt.title('Image')
plt.imshow(image[:,:,[2,1,0]])
plt.show()
