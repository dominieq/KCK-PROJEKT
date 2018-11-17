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

def display_image(title, img):
		cv2.imshow(title, img)
		cv2.waitKey(0) & 0xFF
		cv2.destroyAllWindows()

def adjust_photo(i):
	if i < 10:
		file = "Resources/nutki_0"+repr(i)+".JPG"
	else:
		file = "Resources/nutki_"+repr(i)+".JPG"

	# load the image and resize it in order to
	# increase accuracy of edge detection and speed up image processing
	image = cv2.imread(file)
	ratio = image.shape[0] / 500.0  #it will allow us to work later on the original image
	orig = image.copy()
	image = imutils.resize(image, height = 500)

	# convert the image to grayscale, blur it, and find edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0) #remove high frequency noise
	edged = cv2.Canny(gray, 75, 200)

	# find the largest contours
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1] #OpenCV 2.4 and OpenCV 3 return contours differently
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

	# loop over the contours
	for c in cnts:
		#check the length of the contour if it is closed
		perimeter = cv2.arcLength(c, True) 
		approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
	#four points found? assume it's our sheet of paper
		if len(approx) == 4:
			screenCnt = approx
			break

	#turn the sheet of paper using original image
	warped = extract_sheet(orig, screenCnt.reshape(4, 2) * ratio)

	height, width, _= warped.shape
	if height < width:
		directory = "turned"
	else:
		directory = "output"
	# convert the warped image to grayscale, then threshold it
	# to give it that 'black and white' paper effect
	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	warped = imutils.resize(warped, width=1000)
	cv2.imwrite(directory+"/warped"+repr(i)+"_gray.jpg", warped)
		
	return warped
