import cv2
import numpy as np
import sys, os
import staffs as sts
import blob_detector as bd
import adjuster


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


def writeName(img, places, photoId, whatType):
    font, scale, color, typ = [cv2.FONT_HERSHEY_SIMPLEX, 1, (255,153,0), 2]
    for placeId in range(len(places)):
        if whatType[placeId]:
            cv2.putText(img, 'W', places[placeId], font, scale, color, typ)
        else:
            cv2.putText(img, 'B', places[placeId], font, scale, color, typ)
    cv2.imwrite("keys/image_" + repr(photoId) + ".jpg", img)


def findKey(yHigh, yLow, x, y, img):
    
    propHigh, propLow = [int(y * 0.007), int(y * 0.02)]
    xLeft, xRight = [int(x * 0.07), int(x * 0.125)]
    found = False

    for i in range(yLow + propHigh, yLow + propLow):
        for j in range(xLeft, xRight):
            if not img[i, j].any():  
                found = True
                break    
        if found:
            break
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return found, (xLeft, yHigh - propHigh)



def processImage(index, img, staves, result):
    places, whatKey = [[], []]
    try:
        len(staves)
        img.shape        
    except:
        print("    [keys.py] Can't open file number " + repr(index))
    else:
        # print("    [keys.py] Plik " + repr(index) + " ma " + repr(len(staves)) + " pięciolini")
        args = list(img.shape)
        height, width, howManyKeys = [args[0], args[1], 0]

        for stave in staves:
            isKey, place = findKey(stave.min_range, stave.max_range, width, height, img)

            places.append(place)
            whatKey.append(isKey)

            if isKey : howManyKeys = howManyKeys + 1 
        
        # print("        [keys.py] Znaleziono " + repr(howManyKeys) + " kluczy")
        writeName(result, places, index, whatKey)
        

def lookForKeys():
    for i in range(1, 43):
        if i in [1, 2, 3, 7, 11, 15, 29, 32, 39]:
            continue
        else:
            # print("[keys.py] Rozpoczynam wyszukiwanie w obrazie " + repr(i))
            processImage(i)
    
        
   