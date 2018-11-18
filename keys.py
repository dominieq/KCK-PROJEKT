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
    font, scale, color, typ = [cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2]
    for placeId in range(len(places)):
        if whatType[placeId]:
            cv2.putText(img, 'Klucz wiolinowy', places[placeId], font, scale, color, typ)
        else:
            cv2.putText(img, 'Klucz basowy', places[placeId], font, scale, color, typ)
    cv2.imwrite("keys/image_" + repr(photoId) + ".jpg", img)


""" Finds a key on staves

:param yHigh - stave.min_range
:param yLow - stave.max_range
:param x - width of an image
:param y - height of an image
:param img - processed image
"""
def findKey(yHigh, yLow, x, y, img):
    """
    :prop - an approximate value to add or subtract from yHigh or yLow
        to display only keys
    :xLeft - the left border of a field where we are searching for a key
    :xRight - the right border of a field where we are searching for a key
    """ 
    propHigh, propLow, xLeft, xRight = [int(y * 0.007), int(y * 0.02), int(x * 0.07), int(x * 0.125)]
    found = False

    for i in range(yLow + propHigh, yLow + propLow):
        for j in range(xLeft, xRight):
            if not img[i, j].any():  
                found = True
                break    
        if found:
            break
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return found, (xLeft, yHigh - propLow)


def processImage(index, imagePath, stavesPath):

    stavesImage = adjuster.adjust_photo(index)
    blockPrint()
    staves = sts.get_staffs(stavesImage, index)
    enablePrint()
    img = cv2.imread(imagePath, 0)
    places = []
    whatKey = []

    try:
        len(staves)
        img.shape        
    except:
        print("    Can't open file number " + repr(index))
    else:
        print("    Plik " + repr(index) + " ma " + repr(len(staves)) + " pięciolini")
        args = list(img.shape)
        height, width, howManyKeys = [args[0], args[1], 0]

        for stave in staves:
            isKey, place = findKey(stave.min_range, stave.max_range, width, height, img)
            places.append(place)
            whatKey.append(isKey)
            if isKey : howManyKeys = howManyKeys + 1 
        
        print("        Znaleziono " + repr(howManyKeys) + " kluczy")
        writeName(img, places, index, whatKey)
        


for i in range(1, 31):
    if i in [1, 2, 3, 7, 11, 12, 14, 15, 29]:
        continue
    else:
        print("Rozpoczynam wyszukiwanie w obrazie " + repr(i))
        processImage(i, "staffs/staffs" + repr(i) + "_erode.png", "output/warped" + repr(i) + "_gray.jpg")
    
        
   