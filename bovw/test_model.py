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

# Custom
from kmeans import kmeans
from write_file import exportSIFT,exportKMEANS
from load_images import load_images_from_folder
from load_features import importSIFT, checkDataset
from image_class import calculate_centroids_histogram
from define_class import defineClass
import pysift

def testModel(test_images, clf,model, k=100, train_number="1"):
    print("SIFT - PROCESS")
    # Takes the descriptor list which is unordered one
    descriptor_list = [] 

    # Takes the sift features that is seperated class by class for train data
    shiftData = {}

    brisk = cv2.BRISK_create(30)
    for image in test_images:
        print("-> Image : " + image)

        #kp, des = pysift.computeKeypointsAndDescriptors(images[image]) 
        if checkDataset("bovw/results/SIFT/descriptors", image):
            kp,des = importSIFT('bovw/results/SIFT',image)
        else:
            start_time = time.time()
            print("-> Computing SIFT")
            #kp, des = pysift.computeKeypointsAndDescriptors(test_images[image])
            kp, des = brisk.detectAndCompute(test_images[image], None)
            print("\n \t EXPORT SIFT : " + image)
            exportSIFT(kp, des,'bovw/results/SIFT',image)
            print("\t -> filename  : " + os.path.splitext(image)[0] +".txt")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("\t Done :" + str(elapsed_time)[0:4] +"s (" + str(elapsed_time/60)[0:4] +"min) \n")

        shiftData[image] = des 
        descriptor_list.extend(des) 

        print("-> DES : " + str(len(des)))
        print("\n")
    
    print("Done \n")

    #print("KMEANS")
    # Takes the central points which is visual words    
    #_,all_visual_words, KmeansModel = kmeans(k, descriptor_list) 
    #print("-> visual_words : " + str(len(all_visual_words)))
    #print("Done\n")

    print("HISTOGRAMS")
    # Creates histograms for train data
    test_class,test_featvec = calculate_centroids_histogram(test_images, shiftData, model, k)
    print("-> train_class       : " + str(len(test_class)) + " classes")
    print("-> train_featvec     : " + str(len(test_featvec)) + " histograms")
    print("Done\n")

    print("TESTING SVM")
    prediction = clf.predict(test_featvec)
    labels = clf.classes_
    print("Done\n")

    return prediction, test_class, labels