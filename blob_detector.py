import cv2
import numpy as np
from skimage.filters import threshold_local
from config import *


def display_image(title, img):
		cv2.imshow(title, img)
		cv2.waitKey(0) & 0xFF
		cv2.destroyAllWindows()

def remove_lines(im_with_blobs, method):
    T = threshold_local(im_with_blobs, 11, offset = 20, method = method)#generic, mean, median
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
    # T = threshold_local(im_with_blobs, 11, offset = 20, method = "median")#generic, mean, median
    # im_with_blobs = (im_with_blobs > T).astype("uint8") * 255

    # im_inv = (255 - im_with_blobs)
    # kernel = cv2.getStructuringElement(ksize=(1, int(im_inv.shape[0] / 500)), shape=cv2.MORPH_RECT)
    # horizontal_lines = cv2.morphologyEx(im_inv, cv2.MORPH_OPEN, kernel)
    # horizontal_lines = (255 - horizontal_lines)

    hl1 = remove_lines(im_with_blobs, "median")
    hl2 = remove_lines(im_with_blobs, "mean")
    hl3 = remove_lines(im_with_blobs, "gaussian")
    # horizontal_lines = cv2.erode(horizontal_lines, np.ones((1, 1)))

    return hl1, hl2, hl3
    # return im_with_blobs, horizontal_lines

    # cv2.imwrite("output/8a_lines_horizontal_removed.png", horizontal_lines)
    #     cv2.imwrite("output/8a_lines_vertical_removed.png", vertical_lines)

    # im_with_blobs = vertical_lines
    # im_with_blobs = cv2.cvtColor(im_with_blobs, cv2.COLOR_GRAY2BGR)

    # # Set up the SimpleBlobDetector with default parameters.
    # params = cv2.SimpleBlobDetector_Params()
    # params.filterByArea = True
    # params.minArea = 225
    # params.maxArea = 1500
    # params.filterByCircularity = True
    # params.minCircularity = 0.6
    # params.filterByConvexity = True
    # params.minConvexity = 0.9
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01

    # detector = cv2.SimpleBlobDetector_create(params)
    # keypoints = detector.detect(im_with_blobs)

    # cv2.drawKeypoints(im_with_blobs, keypoints=keypoints, outImage=im_with_blobs, color=(0, 0, 255),
    #                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # if SAVING_IMAGES_STEPS:
    #     cv2.imwrite("output/8b_with_blobs.jpg", im_with_blobs)

    # '''
    # Here we enumerate notes.
    # '''
    # staff_diff = 3 / 5 * (staffs[0].max_range - staffs[0].min_range)
    # bins = [x for sublist in [[staff.min_range - staff_diff, staff.max_range + staff_diff] for staff in staffs] for x in
    #         sublist]

    # keypoints_staff = np.digitize([key.pt[1] for key in keypoints], bins)
    # sorted_notes = sorted(list(zip(keypoints, keypoints_staff)), key=lambda tup: (tup[1], tup[0].pt[0]))

    # im_with_numbers = im_with_blobs.copy()

    # for idx, tup in enumerate(sorted_notes):
    #     cv2.putText(im_with_numbers, str(idx), (int(tup[0].pt[0]), int(tup[0].pt[1])),
    #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                 fontScale=1, color=(255, 0, 0))
    #     cv2.putText(im_with_blobs, str(tup[1]), (int(tup[0].pt[0]), int(tup[0].pt[1])),
    #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                 fontScale=1, color=(255, 0, 0))
    # if SAVING_IMAGES_STEPS:
    #     cv2.imwrite("output/8c_with_numbers.jpg", im_with_numbers)
    #     cv2.imwrite("output/8d_with_staff_numbers.jpg", im_with_blobs)

    # if VERBOSE:
    #     print("Keypoints length : " + str(len(keypoints)))

    # return sorted_notes