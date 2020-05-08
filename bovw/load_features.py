import os
import numpy as np
from sklearn.externals import joblib
from os import walk

def importSIFT(folder, filename):
    descriptorsFile = folder + "/descriptors/" + os.path.splitext(filename)[0] + ".txt"
    des = np.genfromtxt(descriptorsFile)

    keypointsFile = folder + "/keypoints/" + os.path.splitext(filename)[0]  + ".txt"
    key = np.genfromtxt(keypointsFile)
    #key = []

    return key,des

def checkDataset(folder,filename):
    return os.path.exists(folder + "/" +os.path.splitext(filename)[0] +".txt")

def checkKmeans(k, folder, train_number="1"):
    return ((os.path.exists(folder + "/clusters/KMEANS_TRAIN" + train_number +"_" + str(k) +".txt")) & (os.path.exists(folder + "/models/model_KMEANS_TRAIN" + train_number +"_" + str(k) +".pkl")))

def importKmeans(k,folder, train_number="1"):
    clusterFile = folder + "/clusters/KMEANS_TRAIN"+ train_number + "_" + str(k) +".txt"
    cluster = np.genfromtxt(clusterFile)

    print("SAVING KMEANS MODEL")
    model = joblib.load("bovw/results/KMEANS/models/model_KMEANS_TRAIN" + train_number +"_" + str(k) +".pkl")
    print("Done\n")

    return cluster, model
