import os
import numpy as np

import cv2
import time
import math

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Perform k-means clustering and vector quantization
from scipy.cluster.vq import kmeans, vq
from sklearn import svm
from sklearn import preprocessing

# Custom
from kmeans import kmeans
from write_file import exportSIFT,exportKMEANS
from load_images import load_images_from_folder
from load_features import importSIFT, checkDataset
from image_class import calculate_centroids_histogram
from define_class import defineClass
from utils import *
import pysift

def testModel(test_images, allClassifier,model, k=100, train_number="1"):
    print("SIFT - PROCESS")
    #Start timing
    start_time = time.time()

    # Takes the descriptor list which is unordered one
    descriptor_list = [] 

    # Takes the sift features that is seperated class by class for train data
    shiftData = {}

    # SIFT remplacement
    brisk = cv2.BRISK_create(30)

    # Console message
    status = ""
    for image in test_images:

        #kp, des = pysift.computeKeypointsAndDescriptors(images[image]) 
        if checkDataset("bovw/results/SIFT/descriptors", image):
            start_timeSIFT = time.time()
            _,des = importSIFT('bovw/results/SIFT',image)

            end_timeSIFT = time.time()
            elapsed_timeSIFT = end_timeSIFT - start_timeSIFT
            status = "LOADING SIFT :" + str(elapsed_timeSIFT)[0:4] +"s (" + str(elapsed_timeSIFT/60)[0:4] +"min)"
        else:
            start_timeSIFT = time.time()
            kp, des = brisk.detectAndCompute(test_images[image], None)
            exportSIFT(kp, des,'bovw/results/SIFT',image)

            end_timeSIFT = time.time()
            elapsed_timeSIFT = end_timeSIFT - start_timeSIFT
            status = "EXPORT SIFT :" + str(elapsed_timeSIFT)[0:4] +"s (" + str(elapsed_timeSIFT/60)[0:4] +"min)"

        shiftData[image] = des 
        descriptor_list.extend(des) 
        printProgressBar(len(shiftData), len(test_images), image)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("-> Total  : " + str(len(descriptor_list)) + " descriptors")
    print("Done :" + str(elapsed_time)[0:4] +"s (" + str(elapsed_time/60)[0:4] +"min) \n")

    # Deleting all images in test_images dict : release memory
    train_images = {image : [] for image in test_images}

    print("HISTOGRAMS")
    #Start timing
    start_time = time.time()

    # Creates histograms for train data
    test_class,test_featvec = calculate_centroids_histogram(test_images, shiftData, model, k)
    print("-> test_class       : " + str(len(test_class)) + " classes")
    print("-> train_featvec     : " + str(len(test_featvec)) + " histograms")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Done :" + str(elapsed_time)[0:4] +"s (" + str(elapsed_time/60)[0:4] +"min) \n")

    print("TESTING CLASSFIERS")
    predictions = {}
    labels = []

    for i,clf in enumerate(allClassifier):
        #print(clf)
        predictions[clf] = allClassifier[clf].predict(test_featvec)
        labels = allClassifier[clf].classes_
        printProgressBar(i+1, len(allClassifier), clf)
    
    print("Done\n")
    return predictions, test_class, labels