import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
'''
ARUCO_DICT = {
	"DICT_4X4_50": cv.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
}
'''

dicti = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

image = cv.imread('Room with ArUco Markers-20240324/20221115_113412.jpg')
'''plt.title('Image')
plt.imshow(image[:,:,[2,1,0]])
plt.show()
'''
poster_image = cv.imread('ronaldo1.jpeg')
'''plt.title('Poster Image')
plt.imshow(poster_image[:,:,[2,1,0]])
plt.show()'''

coord_poster = np.array([[0, 0], [poster_image.shape[1], 0], [poster_image.shape[1], poster_image.shape[0]], [0, poster_image.shape[0]]], dtype=np.float32)

parameters = cv.aruco.DetectorParameters()
corners, ids, rejected = cv.aruco.detectMarkers(image, dicti, parameters=parameters)

corn_cord = np.array(corners[0][0], dtype=np.float32)
print(corn_cord)
print(coord_poster)

M = cv.getPerspectiveTransform(coord_poster, corn_cord)
print(M)
print(corners)
print(ids)
print(rejected)
print(f'Corners: {corners[0][0]}')

M[0,0]*=5
M[1,1]*=5
#plt.scatter(corners)
img_mod = image.copy()

poster_trans = cv.warpPerspective(poster_image,M,(image.shape[1],image.shape[0]))
cv.imshow('Trans Image',poster_trans)
plt.title('Transformed Poster')
plt.imshow(poster_trans[:,:,[2,1,0]])
plt.show()

poster_gray = cv.cvtColor(poster_trans, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(poster_gray, 1, 255, cv.THRESH_BINARY)
mask = cv.bitwise_not(mask)
result_image = cv.bitwise_and(image, image, mask=mask)

final_result = cv.add(result_image, poster_trans)
plt.title('Transformed Poster')
plt.imshow(final_result[:,:,[2,1,0]])
plt.show()
