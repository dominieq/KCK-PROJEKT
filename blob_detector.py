import cv2
import numpy as np
from skimage.filters import threshold_local
from config import *
import imutils

def display_image(title, img):
		cv2.imshow(title, img)
		cv2.waitKey(0) & 0xFF
		cv2.destroyAllWindows()

# def enumerate_notes(im_with_blobs, keypoints, staffs):
#     staff_diff = 3/5 * (staffs[0].max_range - staffs[0].min_range)
#     # lista dwuelementowych list zawierających [min-diff, max+diff] - poszerzamy pięciolinię
#     staffs_coordinates = []
#     for staff in staffs:
#         staffs_coordinates.append(staff.min_range - staff_diff)
#         staffs_coordinates.append(staff.max_range + staff_diff)

#     keypoints_staff = np.digitize([key.pt[1] for key in keypoints], staffs_coordinates)
#     keypoints_staff = [int((key+1)/2) for key in keypoints_staff]

#     sorted_notes = sorted(list(zip(keypoints, keypoints_staff)), key=lambda tup: (tup[1], tup[0].pt[0]))

#     im_with_numbers = im_with_blobs.copy()

#     for idx, tup in enumerate(sorted_notes):
#         cv2.putText(im_with_numbers, str(idx), (int(tup[0].pt[0]), int(tup[0].pt[1])),
#                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                     fontScale=1, color=(255, 0, 0))
#         cv2.putText(im_with_blobs, str(tup[1]), (int(tup[0].pt[0]), int(tup[0].pt[1])),
#                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                     fontScale=1, color=(255, 0, 0))
#     return im_with_numbers, im_with_blobs, sorted_notes

# def find_blobs(image):
#     # Set up the SimpleBlobDetector with default parameters.
#     params = cv2.SimpleBlobDetector_Params()
#     params.minThreshold = 10
#     params.maxThreshold = 3000
#     params.minDistBetweenBlobs = 0
#     params.filterByArea = True
#     params.minArea = 250
#     params.maxArea = 700
# ##     whole notes params
# #     params.minArea = 250
# #     params.maxArea = 410
#     params.filterByCircularity = False
# #     params.minCircularity = 0
#     params.filterByConvexity = False
# #     params.minConvexity = 0
#     params.filterByInertia = False
# #     params.minInertiaRatio = 0.01

#     detector = cv2.SimpleBlobDetector_create(params)
#     keypoints = detector.detect(image)

#     cv2.drawKeypoints(image, keypoints=keypoints, outImage=image, color=(0, 0, 255),
#                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     return image, keypoints

def find_contours(original, thresholded, staffs):
    contours = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if imutils.is_cv2() else contours[1] #OpenCV 2.4 and OpenCV 3 return contours differently
    contours_notes = []

    for cnt in contours:
            contour_perimeter = cv2.arcLength(cnt, True)
            #whole note size
            if 50 < contour_perimeter < 80:
                contours_notes.append(cnt)

    centres = []
    for cnt in contours_notes:
        max_x = np.max([x[0][:][0] for x in cnt])
        min_x = np.min([x[0][:][0] for x in cnt])
        max_y = np.max([x[0][:][1] for x in cnt])
        min_y = np.min([x[0][:][1] for x in cnt])
        mid_x = (max_x+min_x)/2
        mid_y = (max_y+min_y)/2
        centres.append([mid_x,mid_y])

    staff_diff = 3/5 * (staffs[0].max_range - staffs[0].min_range)
    staffs_coordinates = []
    for staff in staffs:
        staffs_coordinates.append(staff.min_range - staff_diff)
        staffs_coordinates.append(staff.max_range + staff_diff)
   
    keypoints_staff = np.digitize([center[1] for center in centres], staffs_coordinates)
    keypoints_staff = [int((key+1)/2) for key in keypoints_staff]

    sorted_notes = sorted(list(zip(centres, keypoints_staff)), key=lambda tup: (tup[1], tup[0][0]))

    background =  cv2.cvtColor(original.copy(), cv2.COLOR_GRAY2BGR)

# font, scale, color, typ = [cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2]

    for idx, tup in enumerate(sorted_notes):
        cv2.putText(background, str(tup[1])+"|"+str(idx), (int(tup[0][0]), int(tup[0][1]-15)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0, 0, 255)) #ype = 2)

    with_notes = cv2.drawContours(background, contours_notes, -1, (0,0,255), 2)
    with_notes = imutils.resize(with_notes, height = 1000)
    display_image("jol",with_notes)
    return with_notes


def remove_lines(im_with_blobs, method, size = 11, off = 10):
    T = threshold_local(im_with_blobs, size, offset = off, method = method)#generic, mean, median
    im_with_blobs = (im_with_blobs > T).astype("uint8") * 255

    im_inv = (255 - im_with_blobs)
    kernel = cv2.getStructuringElement(ksize=(1, int(im_inv.shape[0] / 500)), shape=cv2.MORPH_RECT)
    horizontal_lines = cv2.morphologyEx(im_inv, cv2.MORPH_OPEN, kernel)
    horizontal_lines = (255 - horizontal_lines)
    return horizontal_lines



def detect_blobs(input_image, staffs):
    """
    Detects blobs with given parameters.
    https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    """
    im_with_blobs = input_image.copy()

    horizontal_removed = remove_lines(im_with_blobs, "mean", size = 11, off = 20) 
    horizontal_removed_contours = cv2.erode(horizontal_removed, np.ones((7, 5)))    

    with_contours = find_contours(im_with_blobs, horizontal_removed_contours, staffs)

#   better version of cv2.erode parameters for blob detection
    horizontal_removed = cv2.erode(horizontal_removed, np.ones((9, 5)))
    
    im_with_blobs = horizontal_removed

    return horizontal_removed, im_with_blobs
    # im_with_blobs, keypoints = find_blobs(im_with_blobs)
    # im_with_numbers, im_with_blobs, sorted_notes = enumerate_notes(im_with_blobs, keypoints, staffs)
    # print("Keypoints length : " + str(len(keypoints)))
    # return horizontal_removed, im_with_blobs, with_contours


    # return sorted_notes