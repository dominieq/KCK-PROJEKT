from staffs import get_staffs
from blob_detector import *
from adjuster import adjust_photo

def main():
    for i in range(30,32):
        if i not in [7,11,29]:
            adjust_photo(i)
    for i in range(16,32):
        if i not in [7,11,15,29]:
            thresholded_image, staffs = get_staffs("output/warped"+repr(i)+"_gray.jpg", i)
    # detect_blobs(image, staffs)

main()