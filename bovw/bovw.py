import os
import numpy as np

import cv2
import time
import math

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from accuracy import *
#import matplotlib.pyplot as plt

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

print("DATASET CONFIGURATION")

# UTILISEZ UNIQUEMENT LA TRAIN 4 => NOM IMAGE DIFFERENT
train_number = "4"

print("Train dataset n° : " + str(train_number))
print("Done\n")

print("LOADING TRAIN IMAGE ")
train_images = load_images_from_folder("bovw/dataset/train" + train_number)  # take all images category by category 
print("Done \n")

print("LOADING TEST IMAGE ")
test_images = load_images_from_folder("bovw/dataset/query" + train_number)  # take all images category by category 
print("Done \n")

print("TRAINING FROM DATASET")
k = 2000
clf,KmeansModel = trainModel(train_images, k, train_number)
print("Done \n")

print("TESTING QUERY")
prediction, test_class, labelsCLF = testModel(test_images, clf, KmeansModel, k, train_number)
print ("->test_class = "  + str(test_class))
print ("->prediction = "  + str(prediction))
print("Done \n")

print("ACCURACY - CONFUSION MATRIX")
accuracy = accuracy_score(test_class,prediction)
print("->accuracy = " + str(accuracy))

cm = confusion_matrix(test_class, prediction,labels=labelsCLF)
print(cm)
plot_confusion_matrix(cm,labelsCLF,'Confusion Matrix - Art Recognition (' + str(k)+' visuals words)')

file_object = open("bovw/results/ACCURACY/results.csv", "a")
result = "SVM" + "," + str(train_number) + "," + str(k) + "," + str(len(labelsCLF))+ "," +str(accuracy) + "\n" 
file_object.write(result)
file_object.close()
print("Done\n")