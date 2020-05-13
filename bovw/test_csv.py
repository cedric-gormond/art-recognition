from define_class import listArtistCSV, defineClassWithArtists, defineClassWithOnlyFilename
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from load_features import *
from accuracy import *
from load_images import *
import time
import json

print("LOADING TRAIN IMAGE ")
train_images = load_images_from_folder("bovw/dataset/train4")  # take all images category by category 
print("Done \n")

i = 0
for image in train_images:
    
    status = ""
    if checkDataset("bovw/results/SIFT/descriptors", image):
        start_timeSIFT = time.time()
        #print("\r \n -> Computing SIFT \r")
        #print("\r \n \t EXPORT SIFT : " + image + "\r")

        end_timeSIFT = time.time()
        elapsed_timeSIFT = end_timeSIFT - start_timeSIFT
        #print("\r \t Done :" + str(elapsed_timeSIFT)[0:4] +"s (" + str(elapsed_timeSIFT/60)[0:4] +"min) \n \r")
        status = "EXPORT SIFT :" + str(elapsed_timeSIFT)[0:4] +"s (" + str(elapsed_timeSIFT/60)[0:4] +"min)"

    i=i+1
    printProgressBar(i, len(train_images), image, status)
    