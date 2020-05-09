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

for image in train_images:
    print("-> Image : " + image)

    if checkDataset("bovw/results/SIFT/descriptors", image):
        start_time = time.time()

        descriptorsFile = "bovw/results/SIFT/descriptors/" + os.path.splitext(image)[0] + ".txt"
        #des = np.genfromtxt(descriptorsFile, dtype="i8")
        #des = list()
        des = [ list(map(int, line.rstrip('\n').split())) for line in open(descriptorsFile)]

        end_time = time.time()
        print("TIME : " + str(end_time - start_time))
        input()
        print(des)


