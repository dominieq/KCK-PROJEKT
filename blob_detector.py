import cv2
import numpy as np
from skimage.filters import threshold_local
from config import *
import imutils

def display_image(title, img):
		cv2.imshow(title, img)
		cv2.waitKey(0) & 0xFF
		cv2.destroyAllWindows()
def add_note_height(sorted_notes, staffs):
    #([x,y,type],staff)
    extended_notes = []
    for note in sorted_notes:
        if note[1]!=0:
            diff = staffs[note[1]-1].max_range - staffs[note[1]-1].min_range
            staff_top = staffs[note[1]-1].min_range - 0.1*diff
            staff_bottom = staffs[note[1]-1].max_range + 0.1*diff
            lines_space = (staff_bottom - staff_top)/4  
            if note[0][1] >= staff_bottom:
                print(note[0][1])
                print("Bottom: "+str(staff_bottom))
            elif  note[0][1] <= staff_top:
                print(note[0][1])
                print("top: "+str(staff_top))
            detected = False
            for i in range(0,4):
                if (staff_bottom - i * lines_space) > note[0][1] > (staff_bottom - (i+1) * lines_space):
                    new_note = ([note[0][0],note[0][1],note[0][2]], note[1], int(i+1))
                    extended_notes.append(new_note)
                    detected = True
            if not detected:
                new_note = ([note[0][0],note[0][1],note[0][2]], note[1], 0)
                extended_notes.append(new_note)
    return extended_notes

def choose_note_color(note, staff_num):
        if note[0][2] == 0:
            color = (0,0,255) #red
        elif note[0][2] == 1:
            color = (255,0,255) #magenta
        elif note[0][2] == 2:
            color = (255,0,0) #blue         
        else:
            color = (0,102,255) #orange
        return color

def minims_and_quarters(original_image, contours):
    '''
    function divide minim_or_quarter list (contours argument) into minims_list and quarters_list 
    '''
    T = threshold_local(original_image, 11, offset = 12, method = "mean")
    light_threshold = (original_image > T).astype("uint8") * 255
    # light_threshold = cv2.erode(light_threshold, np.ones((1, 2)))
    minims = []
    quarters = []
    for cnt in contours:
        max_x = np.max([x[0][:][0] for x in cnt]) 
        min_x = np.min([x[0][:][0] for x in cnt]) 
        max_y = np.max([x[0][:][1] for x in cnt]) 
        min_y = np.min([x[0][:][1] for x in cnt]) 
        x_diff = max_x - min_x
        y_diff = max_y - min_y
        x1 = int(min_x + 0.2 * x_diff)
        x2 = int(max_x - 0.3 * x_diff)
        y1 = int(min_y + 0.55 * y_diff)
        y2 = int(max_y - 0.1 * y_diff)

        '''
        find mean of note's inside
        '''
        mean = np.mean(light_threshold[y1:y2, x1:x2])
        if mean > 140:
            minims.append(cnt)
        else:
            quarters.append(cnt)
        
    return minims, quarters


def find_centres(contours, note_type):
    centres = []
    x_cent = 0 
    y_cent = 0
    for contour in contours:
        max_x, min_x, max_y, min_y = find_boundaries(contour)
        if note_type == 0:
            x_cent = (max_x + min_x)/2
            y_cent = (max_y + min_y)/2
        else: 
            y_cent = min_y + 0.7 * (max_y - min_y)
            if note_type in [1,2]:
                x_cent = min_x + 1/2 * (max_x - min_x)
            elif note_type == 3:            
                x_cent = min_x + 1/4 * (max_x - min_x)
            else:
                print("note_type incorrect value: "+repr(note_type))
                continue
        centres.append([x_cent, y_cent, note_type])
    return centres

def find_boundaries(contour):
    max_x = np.max([x[0][:][0] for x in contour])
    min_x = np.min([x[0][:][0] for x in contour])
    max_y = np.max([y[0][:][1] for y in contour])
    min_y = np.min([y[0][:][1] for y in contour])
    return max_x, min_x, max_y, min_y

def extract_type_of_notes(contours, min_size, max_size, note_type):
    '''
    look for note center and append note type to its x and y 
    '''
    notes_contours = []
    for contour in contours:
        max_x, min_x, max_y, min_y = find_boundaries(contour)
        '''
        min_x>110 - cut detected keys
        (max_x-min_x)>10 and (max_y-min_y)>10 - remove noise
        '''
        if min_x > 110 and (max_x-min_x) > 10 and (max_y-min_y) > 10:
            contour_perimeter = cv2.arcLength(contour, True)
            if min_size <= contour_perimeter <= max_size:
                notes_contours.append(contour)
    return notes_contours

def find_contours(original, thresholded, staffs):
    contours = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    '''
    OpenCV 2.4 and OpenCV 3 return contours different way
    '''
    contours = contours[0] if imutils.is_cv2() else contours[1] 

    full_notes = extract_type_of_notes(contours, 55, 90, 0)
    centres_full = find_centres(full_notes, 0)
    '''
    eight notes, minims and quarters find by their cotnour perimeter and then width
    '''
    eight_notes = extract_type_of_notes(contours, 110, 170, 3)
    eight_notes = [cnt for cnt in eight_notes if ( np.max([x[0][:][0] for x in cnt]) - np.min([x[0][:][0] for x in cnt]) ) > 26 ]
    centres_eights = find_centres(eight_notes, 3)

    minim_or_quarter = extract_type_of_notes(contours, 91, 170, 4)
    minim_or_quarter = [cnt for cnt in minim_or_quarter if ( np.max([x[0][:][0] for x in cnt]) - np.min([x[0][:][0] for x in cnt]) ) < 27 ]
    minim_notes, quarter_notes = minims_and_quarters(original, minim_or_quarter)
    centres_mins = find_centres(minim_notes, 1)
    centres_quarts = find_centres(quarter_notes, 2)
    centres_all = centres_full + centres_mins + centres_eights + centres_quarts
    '''
    make staff borders wider in case of notes laying under or over the staff
    '''
    staff_diff = 3/5 * (staffs[0].max_range - staffs[0].min_range)
    staffs_coordinates = []
    for staff in staffs:
        staffs_coordinates.append(staff.min_range - staff_diff)
        staffs_coordinates.append(staff.max_range + staff_diff)

    '''
    for each note find its staff
    '''
    keypoints_staff = np.digitize([center[1] for center in centres_all], staffs_coordinates)
    keypoints_staff = [int((key+1)/2) for key in keypoints_staff]
    
    sorted_notes = sorted(list(zip(centres_all, keypoints_staff)), key=lambda note: (note[1], note[0][0]))
    sorted_notes = add_note_height(sorted_notes, staffs)
    
    background =  cv2.cvtColor(original.copy(), cv2.COLOR_GRAY2BGR)

    '''
    tupple from 'sorted_notes' structure: ( [x_centre , y_centre , type], staff_num )
    '''
    staff_num = 0
    diff = 0 #in case of changing staff subtract from current index the greatest index from previous staff 
    for idx, note in enumerate(sorted_notes):
        #set note num
        if note[1] != staff_num:
            staff_num = note[1]
            diff = idx - 1
        color = choose_note_color(note, note[1])
        distance = 50
        # if idx % 2 == 1:
        #     distance = distance - 10  
        cv2.putText(background, str(note[0][2])+"|"+str(idx-diff)+"|"+str(note[2])+".", (int(note[0][0]-10), int(note[0][1]-distance)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.3, color=color)

    with_notes = cv2.drawContours(background, full_notes, -1, (0,0,255), 2) # red
    with_notes = cv2.drawContours(with_notes, eight_notes, -1, (0,102,255), 2) #orange
    with_notes = cv2.drawContours(with_notes, minim_notes, -1, (255,0, 255), 2)   #magenta
    with_notes = cv2.drawContours(with_notes, quarter_notes, -1, (255,0,0), 2) #blue   

    return with_notes

def remove_lines(im_with_blobs, method, size = 11, off = 10):
    T = threshold_local(im_with_blobs, size, offset = off, method = method)
    im_with_blobs = (im_with_blobs > T).astype("uint8") * 255

    height, _ = im_with_blobs.shape
    '''
    remove additional words under the staffs
    '''
    for i in range(int(0.90 * height), height):
        im_with_blobs[i][:]=255

    im_inv = (255 - im_with_blobs)
    kernel = cv2.getStructuringElement(ksize=(1, int(im_inv.shape[0] / 650)), shape=cv2.MORPH_RECT)
    horizontal_lines = cv2.morphologyEx(im_inv, cv2.MORPH_OPEN, kernel)
    horizontal_lines = (255 - horizontal_lines)

    return horizontal_lines


def detect_blobs(input_image, staffs):    
    im_with_blobs = input_image.copy()

    horizontal_removed = remove_lines(im_with_blobs, "mean", size = 11, off = 24) 
    # horizontal_removed = cv2.erode(horizontal_removed, np.ones((9, 5)))
    horizontal_removed = cv2.erode(horizontal_removed, np.ones((9, 5)))


    with_contours = find_contours(im_with_blobs, horizontal_removed, staffs)
    
    return horizontal_removed, with_contours