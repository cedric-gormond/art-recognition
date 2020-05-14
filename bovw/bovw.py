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
#os.chdir(r"c:\Users\cedri\Documents\workspace\art_recognition")

print("\n ART RECOGNITION - Les BeauxArt'TSE \n")

print("DATASET CONFIGURATION")
train_number      = "4"   # Dataset number (train + query)
K                 = [6000]  # Configure BoVW with multiple K = [k1,k2,...,kn]. If K is unique, please set K=[k]
                          # K = [50,100,500,1000,2000,3000,4000,5000,6000,7000]
testTrainDataset  = False # Configure test images by train images. 
displayFigures    = True # Display confusion matrixe figures.
displayReports    = False # Display reports in console.

# Program
train_number = str(train_number) if type(train_number) is int else train_number
print("-> Train dataset n° : " + str(train_number))
print("-> Clusters         : " + str(K))
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

for k in K:
    print("TRAINING FROM DATASET WITH k=" + str(k))
    allClassifier,KmeansModel = trainModel(train_images, k, train_number)
    print("Done \n")

    print("TESTING QUERY WITH k=" + str(k))
    predictions, test_class, labelsCLF = testModel(test_images, allClassifier, KmeansModel, k, train_number)
    print("Done \n")

    print("ACCURACY - CONFUSION MATRIX")
    accuracies = {}
    for clf in predictions:
        # Accuracy
        accuracy = accuracy_score(test_class,predictions[clf])
        accuracies[clf] = accuracy
        print("CLF : " + clf)
        print("-> accuracy = " + str(accuracy))
        print(classification_report(test_class,predictions[clf]))  if displayReports else None

        # Confusion matrix
        cm = confusion_matrix(test_class, predictions[clf],labels=labelsCLF)
        #print(cm)
        plot_confusion_matrix(cm,labelsCLF,'Art Recognition (' + clf + ', ' + str(k)+' visuals words)') if displayFigures else None
        print("Done\n")

    # TODO : Creer un seul fichier pour sauver les courbes
    print("SAVING RESULTS")

    # File
    result_path = str("RESULTS_TRAIN" + train_number) + ".csv"
    file_object = open("bovw/results/ACCURACY/train" + str(train_number) + "/" + result_path, "a")

    # Wrinting-in
    result = str("TRAIN" + train_number) + ", " + str(testTrainDataset) + ", " +str(k)+ ", " + str(list(accuracies.values())).strip('[]') + "\n" 
    file_object.write(result)
    file_object.close()
    print("Done\n")