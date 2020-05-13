import os
import numpy as np

import cv2
import time
import datetime
import math

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from utils import *
import matplotlib.pyplot as plt


# Perform k-means clustering and vector quantization
from scipy.cluster.vq import vq
from sklearn import svm
from sklearn import preprocessing

# Custom
from kmeans import kmeans
from write_file import exportSIFT,exportKMEANS
from load_images import load_images_from_folder
from load_features import importSIFT, importKmeans, checkDataset, checkKmeans
from image_class import calculate_centroids_histogram, calculate_centroids_histogram_v2, calculate_centroids_histogram_v3
from define_class import defineClass
import pysift

def trainModel(train_images, k=1000, train_number="1"):
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
    status = ''
    for image in train_images:

        if checkDataset("bovw/results/SIFT/descriptors", image):
            start_timeSIFT = time.time()
            _,des = importSIFT('bovw/results/SIFT',image)

            end_timeSIFT = time.time()
            elapsed_timeSIFT = end_timeSIFT - start_timeSIFT
            status = "LOADING SIFT :" + str(elapsed_timeSIFT)[0:4] +"s (" + str(elapsed_timeSIFT/60)[0:4] +"min)"
        else:
            start_timeSIFT = time.time()

            kp, des = brisk.detectAndCompute(train_images[image], None)
            exportSIFT(kp, des,'bovw/results/SIFT',image)

            end_timeSIFT = time.time()
            elapsed_timeSIFT = end_timeSIFT - start_timeSIFT
            status = "EXPORT SIFT :" + str(elapsed_timeSIFT)[0:4] +"s (" + str(elapsed_timeSIFT/60)[0:4] +"min)"
        
        shiftData[image] = des
        descriptor_list.extend(des) 

        printProgressBar(len(shiftData), len(train_images), image, status)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Done :" + str(elapsed_time)[0:4] +"s (" + str(elapsed_time/60)[0:4] +"min) \n")

    # Deleting all images in train_image dict : release memory
    train_images = {image : [] for image in train_images}

    print("KMEANS (" + str(k) +" clusters)")
    if checkKmeans(k, "bovw/results/KMEANS", train_number):
        print("\t LOADING KMEANS")
        all_visual_words,KmeansModel = importKmeans(k,'bovw/results/KMEANS', train_number)
        print("\t Done")
    else:
        start_time = time.time()
        print("-> Computing KMEANS (this may take time ...)")
        _,all_visual_words, KmeansModel = kmeans(k, descriptor_list)
        
        print("\t EXPORT KMEANS")
        exportKMEANS(all_visual_words, KmeansModel, k, 'bovw/results/KMEANS', train_number)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("\t Done :" + str(elapsed_time)[0:4] +"s (" + str(elapsed_time/60)[0:4] +"min) \n")

    print("Done\n")

    print("HISTOGRAMS")
    #Start timing
    start_time = time.time()

    # Creates histograms for train data
    train_class,train_featvec = calculate_centroids_histogram(train_images, shiftData, KmeansModel, k)
    print("-> train_class       : " + str(len(train_class)) + " elements")
    print("-> train_featvec     : " + str(len(train_featvec)) + " histograms")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Done :" + str(elapsed_time)[0:4] +"s (" + str(elapsed_time/60)[0:4] +"min) \n")

    print("CLASSIFICATION")
    allClassifier = {}
    #Encoding string data into float for LinearSVR and SVR
    train_class_float = []

    # Linear Support Vector Classification
    clf_LinearSVC = svm.LinearSVC()
    clf_LinearSVC.fit(train_featvec, train_class)
    allClassifier["LinearSVC"] = clf_LinearSVC

    # Linear Support Vector Reggression
    #clf_LinearSVR = svm.LinearSVR()
    #clf_LinearSVR.fit(train_featvec, train_class_float)
    #allClassfier.append(clf_LinearSVR)

    # Support Vector Classification
    clf_SVC = svm.SVC()
    clf_SVC.fit(train_featvec, train_class)
    allClassifier["SVC"] = clf_SVC

    # Support Vector Reggression
    #clf_SVR = svm.SVR()
    #clf_SVR.fit(train_featvec, train_class_float)
    #allClassfier.append(clf_SVR)

    # KNN
    clf_KNN = KNeighborsClassifier(n_neighbors=len(train_class))
    clf_KNN.fit(train_featvec, train_class)
    allClassifier["KNN"] = clf_KNN

    print("Done\n")

    return allClassifier, KmeansModel
