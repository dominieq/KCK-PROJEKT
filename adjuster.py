from __future__ import print_function
from skimage.filters import threshold_local
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import cv2
import imutils
from scipy.spatial import distance as dist
import argparse
 
 
def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
 
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
 
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def main():
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--image", required = True,
	# 	help = "Path to the image to be scanned")
	# args = vars(ap.parse_args())

	# load the image and compute the ratio of the old height
	# to the new height, clone it, and resize it in order to
	# increase accuracy of edge detection and speed up image processing
	# image = cv2.imread(args["image"])
	for i in [4,5,6,8,9,10,13,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]:
		if i < 19:
			file = "Resources/nutki_0"+repr(i)+".JPG"
		else:
			file = "Resources/nutki_"+repr(i)+".JPG"
		image = cv2.imread(file)
		ratio = image.shape[0] / 500.0  #it will allow us to work later on the original image
		orig = image.copy()
		image = imutils.resize(image, height = 500)
		
		# convert the image to grayscale, blur it, and find edges
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (5, 5), 0) #remove high frequency noise
		edged = cv2.Canny(gray, 75, 200)
		
		#show the original image and the edge detected image
		# print("STEP 1: Edge Detection")
		# cv2.imshow("Image", image)
		# cv2.imshow("Edged", edged)
		# cv2.waitKey(0) & 0xFF
		# cv2.destroyAllWindows()

		# find the contours in the edged image, keeping only the
		# largest ones, and initialize the screen contour
		cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1] #OpenCV 2.4 and OpenCV 3 return contours differently
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
		
		# loop over the contours
		for c in cnts:
			# approximate the contour
			#check the length of the contour if it is closed
			perimeter = cv2.arcLength(c, True) 
			approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
			# if our approximated contour has four points, then we
			# can assume that we have found our screen
			if len(approx) == 4:
				screenCnt = approx
				break
		
		# show the contour (outline) of the piece of paper
		# print("STEP 2: Find contours of paper")
		# try:
		# 	cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
		# except:
		# 	print("Contour not found")
		# cv2.imshow("Outline", image)
		# cv2.waitKey(0) & 0xFF
		# cv2.destroyAllWindows()

		#turn the sheet of paper using original image
		warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
		
		# convert the warped image to grayscale, then threshold it
		# to give it that 'black and white' paper effect
		warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
		warped = imutils.resize(warped, width=1000)
		cv2.imwrite("output/warped"+repr(i)+"_gray.jpg", warped)
		T = threshold_local(warped, 11, offset = 10, method = "gaussian")
		warped = (warped > T).astype("uint8") * 255
		cv2.imwrite("output/warped"+repr(i)+"_thr.jpg", warped)
		# show the original and scanned images
		# print("STEP 3: Apply perspective transform")
		# cv2.imshow("Original", imutils.resize(orig, height = 650))
		# cv2.imshow("Scanned", imutils.resize(warped, height = 650))
		# cv2.waitKey(0) & 0xFF
		# cv2.destroyAllWindows()

main()