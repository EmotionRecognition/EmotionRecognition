import numpy as np
import os
import os.path
from skimage.io import imread, imsave
from skimage import transform
from scipy.misc import imresize

import random as rnd

def fliph(img):
    return np.fliplr(img)
def flipv(img):
    return  np.flipud(img)
def rotate(img,angle):
    return transform.rotate(img,angle)
def zoom(img,size):
    return imresize(img,size)


def shouldDo(chance):
    number = rnd.randint(0,100)
    if number < chance:
        return True
    else:
        return False
def rndAngle():
    return rnd.uniform(0,360)
def rndZoom():
    if rnd.randint(0,1):
        return  rnd.randint(50,99)
    return rnd.randint(101,400)


if __name__ == '__main__':
    imagesPath = ".\images"
    augmentedImagesPath = ".\imagesAugmented"

    flipHChance = 50#procent
    flipVChance = 50

    rotateChance = 50

    zoomChance = 50


    for dirpath, dirnames, filenames in os.walk(imagesPath):
        for filename in  filenames:
            fullImagePath = os.path.join(dirpath, filename)

            image = imread(fullImagePath)
            if(shouldDo(flipHChance)): image = fliph(image)
            if(shouldDo(flipVChance)): image = flipv(image)
            if(shouldDo(rotateChance)): image = rotate(image, rndAngle())
            if(shouldDo(zoomChance)): image = zoom(image, rndZoom())

            imsave(augmentedImagesPath + "\\"+ filename, image)


