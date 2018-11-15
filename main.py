from staffs import get_staffs
from blob_detector import *
from adjuster import adjust_photo
import numpy as np
import imutils


def display_images(title, img):
        img = [imutils.resize(i, height = 600) for i in img]    
        cv2.imshow(title, np.hstack((img[0],img[1], img[2])))
        cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

#  7,11,15 are not adjusted properly
# 29 is not read by staffs.py
def main():
    # for i in range(30,32):
    #     if i not in [7,11,29]:
    #        adjusted_img = adjust_photo(i)
    #        if i != 29:
    #           thresholded_image, staffs = get_staffs(adjusted_img, i)
            #   thresholded_image, staffs = get_staffs("output/warped"+repr(i)+"_gray.jpg", i)
    i = 4
    adjusted_image = cv2.imread("output/warped"+repr(i)+"_gray.jpg", 0)
    # adjust_photo(i)
    staffs = get_staffs(adjusted_image, i)
    jol, elo = detect_blobs(adjusted_image, staffs)
    blobs = [jol, elo, adjusted_image]
    display_images("siema", blobs)

main()