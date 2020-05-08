import os
import numpy as np

import cv2
import time
import math

from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.neighbors import NearestCentroid
from sklearn.externals import joblib
import matplotlib.pyplot as plt

# Perform k-means clustering and vector quantization
from scipy.cluster.vq import kmeans, vq


# Custom
from kmeans import kmeans
from write_file import exportSIFT,exportKMEANS
from load_images import load_images_from_folder
from load_features import importSIFT, checkDataset
from image_class import calculate_centroids_histogram
from define_class import defineClass, listArtistCSV
import pysift

# Jupyter path
os.chdir(r"c:\Users\cedri\Documents\workspace\art_recognition")

# Main
print("LOADING TRAIN IMAGE ")
images = load_images_from_folder("bovw/dataset/query")  # take all images category by category 
print("Done \n")

start_time = time.time()
timeData = {}

print("SIFT - PROCESS")
# Takes the descriptor list which is unordered one
descriptor_list = [] 

# Takes the sift features that is seperated class by class for train data
shiftData = {}

for image in images:
    print("-> Image : " + image)
    
    #kp, des = pysift.computeKeypointsAndDescriptors(images[image]) 
    if checkDataset("bovw/results/SIFT/descriptors", image):
        kp,des = importSIFT('bovw/results/SIFT',image)
    else:
        kp, des = pysift.computeKeypointsAndDescriptors(images[image])
        print("EXPORT SIFT : " + image)
        exportSIFT(kp, des,'bovw/results/SIFT',image)
        print("-> filename  : " + os.path.splitext(image)[0] +".txt")
        print("Done \n")

    shiftData[image] = des 
    descriptor_list.extend(des) 
    print("-> KP  : " + str(len(kp)))
    print("-> DES : " + str(len(des)))
    print("Done \n")


print("KMEANS")
# kmeansData[image][0] : y_kmeans
# kmeansData[image][1] : visual_words
k=30
kmeansData = {}
for image in images:  
    y_kmeans, visual_words, _ = kmeans(k, shiftData[image])
    kmeansData[image] = [y_kmeans, visual_words] 
    print("-> y_kmeans     : " + str(len(y_kmeans)))
    print("-> visual_words : " + str(len(visual_words)))
    print("Done \n")

    print("EXPORT KMEANS")
    exportKMEANS(visual_words,'bovw/results/KMEANS',image)
    print("-> filename  : " + os.path.splitext(image)[0] +".txt")
    print("Done \n")

    timeData[image] = int(math.ceil(time.time() - start_time))

# Takes the central points which is visual words    
_,all_visual_words, KmeansModel = kmeans(k, descriptor_list) 

display = False

if display:
    print("PLOT")
    for image in images:
        # Redefinition des variables
        y_means = kmeansData[image][0]
        visual_words = kmeansData[image][1]
        des = shiftData[image]

        plt.figure(figsize=(12, 12)).suptitle("Visual words (execution time : " + str(timeData[image]) + " s)")

        plt.subplot(231)
        plt.imshow(images[image])
        plt.title("Image : " + image)

        plt.subplot(232)
        plt.scatter(des[:, 0],des[:, 1], c=None, s=20)
        plt.title(str(len(des)) + " descriptors" )

        plt.subplot(233)
        plt.scatter(des[:, 0], des[:, 1], c=y_means, s=20, cmap='viridis')
        plt.title("K-Means (" + str(len(visual_words)) + " clusters)")

        plt.subplot(234)
        plt.scatter(des[:, 0], des[:, 1], c=y_means, s=20, cmap='viridis')
        plt.scatter(visual_words[:, 0], visual_words[:, 1], c='black', s=100, alpha=0.5)
        plt.title("Descriptors and visual words (centroids of each cluster)")

        plt.subplot(235)
        plt.scatter(visual_words[:, 0], visual_words[:, 1], c='black', s=100, alpha=0.5)
        plt.title("Only visual Words (centroids of each cluster)")

        plt.subplot(236)
        plt.hist(visual_words)
        plt.title("Histogram of visual words")

        plt.show()

        print("Done\n")


print("HISTOGRAMS")
# Creates histograms for train data

all_bovw_feature =  shiftData   
#bovw_train = image_class(all_bovw_feature, all_visual_words) 
train_class,train_featvec = calculate_centroids_histogram(images, shiftData, KmeansModel, k)
#train_class_vector = defineClass(images)
print("-> train_class       : " + str(len(train_class)) + " classes")
print("-> train_featvec     : " + str(len(train_featvec)) + " histograms")
print("Done\n")

#print(bovw_train) 
#plt.figure(figsize=(12, 12))
#plt.hist(train_featvec)

#lt.title("Histogram of visual words")
#plt.show()

print("SVM")
clf = svm.SVC()
clf.fit(train_featvec, train_class)
print("Done\n")

print("SAVING SVM")
joblib.dump((clf, train_featvec, train_class, k), "bovw/results/CLF/bovw_SVM.pkl", compress=3)
print("Done\n")

print("KNN")
clf = NearestCentroid()
clf.fit(train_featvec, train_class)
print("Done\n")

print("SAVING KNN")
joblib.dump((clf, train_featvec, train_class, k), "bovw/results/CLF/bovw_KNN.pkl", compress=3)
print("Done\n")


# Train the Linear SVM
#feature_vectors=np.asarray([v for k,v in bovw_train.items()])
#print([v for k,v in bovw_train.items()])
#X = [v for k,v in bovw_train.items()]
#print("Step 3: Training the SVM classifier")
#clf = svm.SVC()
#clf.fit(X, y)

