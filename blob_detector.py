import cv2
import numpy as np
from skimage.filters import threshold_local
from config import *
import imutils

def display_image(title, img):
		cv2.imshow(title, img)
		cv2.waitKey(0) & 0xFF
		cv2.destroyAllWindows()


def extract_type_of_notes(contours, min_size, max_size, note_type):
    notes_contours = []
    centres = []
    for cnt in contours:
        max_x = np.max([x[0][:][0] for x in cnt])
        min_x = np.min([x[0][:][0] for x in cnt])
        max_y = np.max([x[0][:][1] for x in cnt])
        min_y = np.min([x[0][:][1] for x in cnt])
        #min_x>110 - cut detected keys
        #(max_x-min_x)>10 and (max_y-min_y)>10 - remove noise
        if min_x > 110 and (max_x-min_x)>10 and (max_y-min_y)>10:
            contour_perimeter = cv2.arcLength(cnt, True)
            if min_size <= contour_perimeter <= max_size:
                mid_x = (max_x+min_x)/2
                mid_y = (max_y+min_y)/2
                centres.append([mid_x,mid_y, note_type])
                notes_contours.append(cnt)
        
    return notes_contours, centres

def find_contours(original, thresholded, staffs):
    contours = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #OpenCV 2.4 and OpenCV 3 return contours different way
    contours = contours[0] if imutils.is_cv2() else contours[1] 

    full_notes, centres_full = extract_type_of_notes(contours, 55, 90, 0)

    eight_notes, centres_eights = extract_type_of_notes(contours, 110, 170, 3)
    eight_notes = [cnt for cnt in eight_notes if ( np.max([x[0][:][0] for x in cnt]) - np.min([x[0][:][0] for x in cnt]) ) > 26 ]

    half_or_quarter, centres_hoq = extract_type_of_notes(contours, 91, 170, 2)
    half_or_quarter = [cnt for cnt in half_or_quarter if ( np.max([x[0][:][0] for x in cnt]) - np.min([x[0][:][0] for x in cnt]) ) < 27 ]

    staff_diff = 3/5 * (staffs[0].max_range - staffs[0].min_range)
    staffs_coordinates = []
    for staff in staffs:
        staffs_coordinates.append(staff.min_range - staff_diff)
        staffs_coordinates.append(staff.max_range + staff_diff)
   
    keypoints_staff = np.digitize([center[1] for center in centres_full], staffs_coordinates)
    keypoints_staff = [int((key+1)/2) for key in keypoints_staff]

    sorted_notes = sorted(list(zip(centres_full, keypoints_staff)), key=lambda tup: (tup[1], tup[0][0]))

    background =  cv2.cvtColor(original.copy(), cv2.COLOR_GRAY2BGR)

    for idx, tup in enumerate(sorted_notes):
        cv2.putText(background, str(tup[1])+"|"+str(idx)+"|"+str(tup[0][2]), (int(tup[0][0]-20), int(tup[0][1]-20)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0, 0, 255))

    with_notes = cv2.drawContours(background, full_notes, -1, (0,0,255), 2)
    with_notes = cv2.drawContours(with_notes, eight_notes, -1, (0,255,255), 2)   
    with_notes = cv2.drawContours(with_notes, half_or_quarter, -1, (255,0,255), 2)    
    with_notes = imutils.resize(with_notes, height = 1000)
    return with_notes


def remove_lines(im_with_blobs, method, size = 11, off = 10):
    T = threshold_local(im_with_blobs, size, offset = off, method = method)#generic, mean, median
    im_with_blobs = (im_with_blobs > T).astype("uint8") * 255

    height, _ = im_with_blobs.shape

    #remove additional lines under the staffs
    for i in range(int(0.90 * height), height):
        im_with_blobs[i][:]=255

    im_inv = (255 - im_with_blobs)
    kernel = cv2.getStructuringElement(ksize=(1, int(im_inv.shape[0] / 650)), shape=cv2.MORPH_RECT)
    horizontal_lines = cv2.morphologyEx(im_inv, cv2.MORPH_OPEN, kernel)
    horizontal_lines = (255 - horizontal_lines)
    return horizontal_lines


def detect_blobs(input_image, staffs):
   
    '''
    https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    '''
    
    im_with_blobs = input_image.copy()

    horizontal_removed = remove_lines(im_with_blobs, "mean", size = 11, off = 24) 
    horizontal_removed = cv2.erode(horizontal_removed, np.ones((9, 5)))

    # horizontal_removed_contours = cv2.erode(horizontal_removed, np.ones((7, 5)))    

    with_contours = find_contours(im_with_blobs, horizontal_removed, staffs)

    ## better version of cv2.erode parameters for blob detection
    # horizontal_removed = cv2.erode(horizontal_removed, np.ones((9, 5)))
    
    return horizontal_removed, with_contours