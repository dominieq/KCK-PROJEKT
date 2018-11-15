import cv2
import numpy as np

from config import *
from staff import Staff
from skimage.filters import threshold_local

def display_image(title, img):
		cv2.imshow(title, img)
		cv2.waitKey(0) & 0xFF
		cv2.destroyAllWindows()

def preprocess_image(image, i):
    gray = image.copy()
    element = np.ones((1, 2))
    #bright images are processed more accurately with mean method and smaller range
    if (np.mean(gray) < 190):
        T = threshold_local(gray, 15, offset = 8, method = "median")#generic, mean, median, gaussian
    else:
        T = threshold_local(gray, 7, offset = 8, method = "mean")#generic, mean, median, gaussian

    thresholded = (gray > T).astype("uint8") * 255
    cv2.imwrite("staffs/staffs"+repr(i)+"_thr.png", thresholded)

    thresholded = cv2.erode(thresholded, element)
    cv2.imwrite("staffs/staffs"+repr(i)+"_erode.png", thresholded)

    edges = cv2.Canny(thresholded, 10, 100, apertureSize=3)
    cv2.imwrite("staffs/staffs"+repr(i)+"_canny.png", edges)

    return edges, thresholded


def detect_lines(hough, image, nlines, i):
    # all_lines contains line centres (y-value)
    all_lines = set()
    height, width = image.shape
    # convert to color image so that you can see the lines
    lines_image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for result_arr in hough[:nlines]:
        rho = result_arr[0][0]
        theta = result_arr[0][1]
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho
        shape_sum = width + height
        x1 = int(x0 + shape_sum * (-b))
        y1 = int(y0 + shape_sum * a)
        x2 = int(x0 - shape_sum * (-b))
        y2 = int(y0 - shape_sum * a)

        start = (x1, y1)
        end = (x2, y2)
        diff = y2 - y1
        #detect only horizontal lines with small ends height difference
        if abs(diff) < LINE_ENDS_LEVEL_DIFFERENCE:
            all_lines.add(int((start[1] + end[1]) / 2)) 
            cv2.line(lines_image_color, start, end, (0, 0, 255), 2)

    cv2.imwrite("staffs/staffs"+repr(i)+"_lines.png", lines_image_color)

    return all_lines, lines_image_color


def detect_staffs(all_lines):

    staffs = []
    lines = []
    all_lines = sorted(all_lines)
    for current_line in all_lines:
        # If current line is far away from last detected line
        if lines and abs(lines[-1] - current_line) > PERMISSIBLE_LINES_DISTANCE:
            if len(lines) >= 5:
                # > = 5 lines - staff detected
                staffs.append((lines[0], lines[-1]))
            lines.clear()
        lines.append(current_line)

    # Process the last line
    if len(lines) >= 5:
        if abs(lines[-2] - lines[-1]) <= PERMISSIBLE_LINES_DISTANCE:
            staffs.append((lines[0], lines[-1]))
    return staffs


def draw_staffs(image, staffs,i):

    # Draw the staffs
    width = image.shape[1]
    for staff in staffs:
        cv2.line(image, (0, staff[0]), (width, staff[0]), (0, 255, 255), 2)
        cv2.line(image, (0, staff[1]), (width, staff[1]), (0, 255, 255), 2)
    cv2.imwrite("staffs/staffs"+repr(i)+"_done.png", image)


def get_staffs(image,i):
    image = cv2.imread(image, 0)
    try:
        print(np.mean(image))
    except:
        return
    processed_image, thresholded = preprocess_image(image,i)
    # hough = cv2.HoughLines(processed_image, 1, np.pi / 150, 200)
    hough = cv2.HoughLines(processed_image, 1, np.pi / 100, 100)
    # print(hough2.size-hough1.size)
    all_lines, lines_image_color = detect_lines(hough, thresholded, 80, i)
    staffs = detect_staffs(all_lines)
    draw_staffs(lines_image_color, staffs,i)
    return thresholded, [Staff(staff[0], staff[1]) for staff in staffs]

# # for i in [4,5,13,16,24,25]:
# for i in range(1,31):
#     get_staffs("output/warped"+repr(i)+"_gray.jpg",i)
# # for i in range(5):
#     # if i == 4:
#     # get_staffs("output/warped4_gray.jpg",i)