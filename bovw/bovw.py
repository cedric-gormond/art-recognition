import os
import numpy as np

import cv2
import time
import math


from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report
from accuracy import *
import matplotlib.pyplot as plt

# Perform k-means clustering and vector quantization
from sklearn import svm

# Custom
from train_model import trainModel
from test_model import testModel
from load_images import load_images_from_folder
from define_class import * 

# Jupyter path
os.chdir(r"c:\Users\cedri\Documents\workspace\art_recognition")

print("\n ART RECOGNITION - Les BeauxArt'TSE \n")

print("DATASET CONFIGURATION")
train_number      = "5"
k                 = 5000
testTrainDataset  = True

print("Train dataset n° : " + str(train_number))
print("Clusters         : " + str(k))
print("Done\n")

print("LOADING TRAIN" + train_number +" IMAGE ")
train_images = load_images_from_folder("bovw/dataset/train" + train_number)  # take all images category by category
print("Done \n")

print("LOADING TEST" + train_number + " IMAGE ")
test_images = {}
if(testTrainDataset):
    test_images = train_images
else:
    test_images = load_images_from_folder("bovw/dataset/query" + train_number)  # take all images category by category 
print("Done \n")

print("TRAINING FROM DATASET")
allClassifier,KmeansModel = trainModel(train_images, k, train_number)
print("Done \n")

print("TESTING QUERY")
predictions, test_class, labelsCLF = testModel(test_images, allClassifier, KmeansModel, k, train_number)
#print ("->test_class = "  + str(test_class))
#print ("->prediction = "  + str(predictions))
print("Done \n")

print("ACCURACY - CONFUSION MATRIX")
accuracies = {}
for clf in predictions:
    # Accuracy
    accuracy = accuracy_score(test_class,predictions[clf])
    accuracies[clf] = accuracy
    print("CLF : " + clf)
    print("->accuracy = " + str(accuracy))
    print(classification_report(test_class,predictions[clf]))

    # Confusion matrix
    cm = confusion_matrix(test_class, predictions[clf],labels=labelsCLF)
    #print(cm)
    plot_confusion_matrix(cm,labelsCLF,'Art Recognition (' + clf + ', ' + str(k)+' visuals words)')
    print("Done\n")

print("SAVING RESULTS")
for clf in accuracies:
    # File
    result_path = str("RESULTS_TRAIN" + train_number) + "_" + clf + "_" + str(k) + ".csv"
    file_object = open("bovw/results/ACCURACY/" + result_path, "a")

    # Wrinting-in
    result = str("TRAIN" + train_number) + "," + clf + "," + str(testTrainDataset) + "," + str(len(labelsCLF)) + "," + str(k)+ "," +str(accuracies[clf]) + "\n" 
    file_object.write(result)
    file_object.close()
print("Done\n")