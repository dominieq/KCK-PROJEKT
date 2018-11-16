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

# 7,11,15 are not adjusted properly
# 29 is not read by staffs.py
def main():
    # for i in range(30,32):
    #     if i not in [7,11,15,29]:
    #        adjusted_img = adjust_photo(i)
    #        if i != 29:
    #           thresholded_image, staffs = get_staffs(adjusted_img, i)
        for i in range(4,32):
                if i not in [7,11,15,29]:
                        adjusted_image = adjust_photo(i)
                        height, width = adjusted_image.shape
                        if width < height:
                                print(i)
                                staffs = get_staffs(adjusted_image, i)
                                horizontal_removed = detect_blobs(adjusted_image, staffs)
                                # cv2.imwrite("blobs/"+repr(i)+".jpg", np.hstack((jol, elo, siema)))
                                cv2.imwrite("blobs/"+repr(i)+".jpg", horizontal_removed)


main()