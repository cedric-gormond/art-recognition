import os
import numpy as np

import cv2
import time
import math

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from accuracy import *
import matplotlib.pyplot as plt

# Perform k-means clustering and vector quantization
from scipy.cluster.vq import kmeans, vq
from sklearn import svm

# Custom
from train_model import trainModel
from test_model import testModel
from load_images import load_images_from_folder
from define_class import * 

# Jupyter path
os.chdir(r"c:\Users\cedri\Documents\workspace\art_recognition")

#listArtist = listArtistCSV("bovw/dataset/csv/all_data_info.csv")

print("LOADING TRAIN IMAGE ")
train_images = load_images_from_folder("bovw/dataset/train3")  # take all images category by category 
print("Done \n")

print("LOADING TEST IMAGE ")
test_images = load_images_from_folder("bovw/dataset/query3")  # take all images category by category 
print("Done \n")

print("TRAINING FROM DATASET")
k = 150
clf,KmeansModel = trainModel(train_images, k)
print("Done \n")

print("TESTING QUERY")
prediction, test_class = testModel(test_images, clf, KmeansModel, k)
print ("->test_class = "  + str(test_class))
print ("->prediction = "  + str(prediction))
print("Done \n")

print("ACCURACY - CONFUSION MATRIX")
# Set of classes (avoid duplicates)
labels = list(set(test_class))
accuracy = accuracy_score(test_class,prediction, labels)
print("->accuracy = " + str(accuracy))

cm = confusion_matrix(test_class, prediction)
print(cm)
#showconfusionmatrix(cm)
plot_confusion_matrix(cm,labels,'Confusion Matrix - Art Recognition (' + str(k)+' visuals words)')

file_object  = open("bovw/results/ACCURACY/results.txt", "a")
file_object.write("%f\n" % accuracy)
file_object.close()
print("Done\n")