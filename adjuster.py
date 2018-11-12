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

	#choose left and right points separately 
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	
	#sort by y-value : the one with smaller y will be the top left corner
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	
	rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
	(tr, br) = rightMost

	# #we can count the distance between top left corner and right corners
	# #bigger distance will show us the buttom right corner 
	# D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	# (br, tr) = rightMost[np.argsort(D)[::-1], :]

	return np.array([tl, tr, br, bl], dtype="float32")

def extract_sheet(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	#count the top and bottom widht (Pythagorean theorem)
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	#count the hights 
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	#specify new rectangle containing part of image
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	#count the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

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
		print(i)
		if i < 10:
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
		warped = extract_sheet(orig, screenCnt.reshape(4, 2) * ratio)
		
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