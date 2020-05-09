import os
import numpy as np
import cv2
import time
from utils import *

# load_images_from_folder: takes all images and convert them to grayscale. 
#
# @params : folder path
# @return : a dictionary that holds all images by category. 
def load_images_from_folder(folder):
    images = {}
    
    for filename in os.listdir(folder):  
        category = []
        path = folder + "/" + filename
        
        img = cv2.imread(path,0)
        #print("-> Image : " + filename)

        if img is not None:
            images[filename] = img
        
        printProgressBar (len(images), len([name for name in os.listdir(folder)]), prefix = filename)

    return images