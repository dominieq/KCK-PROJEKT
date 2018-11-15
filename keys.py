import cv2
import staffs as sts
import staff as st
from skimage import data, io, filters, exposure


def findKey(title, yHigh, yLow, x, y):
    prop, xLeft, xRight = [int(y * 0.02), int(width * 0.07), int(width * 0.125)]
    found = False
    for i in range(yHigh - prop, yHigh):
        for j in range(xLeft, xRight):
            if imgEx[i, j].any():
                found = True
                sts.display_image("proba", imgEx[yHigh - prop : yLow + prop, xLeft : xRight ])
                break
        if found:
            break


for i in range(1, 31):
    imgEx = cv2.imread("output/warped" + repr(i) + "_thr_median.jpg", 0)
    staves = sts.get_staffs("output/warped" + repr(i) + "_gray.jpg", i)
    try:
        len(staves)        
    except:
        continue
        #print("Can't open file number " + repr(i))
    else:
        print("Plik " + repr(i) + " ma " + repr(len(staves)) + " pięciolini")
        args = list(imgEx.shape)
        height, width = args[0], args[1]
        #print(height, width)
        for stave in staves:
            #print("    Min: " + repr(stave.min_range))
            #print("    Max: " + repr(stave.max_range) + "\n")
            findKey("image " + repr(i), stave.min_range, stave.max_range, width, height)
        #sts.display_image("image", imgEx)
        
   