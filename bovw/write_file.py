import os
import json
import numpy as np
from sklearn.externals import joblib

def exportSIFT(keypoints, descriptors, folder, filename):
    # Descriptors
    if not os.path.exists(folder + "/descriptors/"):
        os.makedirs(folder + "/descriptors/")
    
    descriptorsFile = folder + "/descriptors/" + os.path.splitext(filename)[0] + ".txt"
    np.savetxt(descriptorsFile, descriptors, fmt="%s")

    # Keypoints
    if not os.path.exists(folder + "/keypoints/"):
        os.makedirs(folder + "/keypoints/")

    keypointsFile = folder + "/keypoints/" + os.path.splitext(filename)[0]  + ".txt"
    np.savetxt(keypointsFile, keypoints, fmt="%s")
    #with open(keypointsFile, 'w') as file:
    #    # Descriptors 
    #    for key in keypoints:
    #        file.write('%s\n' % key)


def exportKMEANS(clusters, model, k, folder):
    kmeansFile = folder + "/clusters/KMEANS_TRAIN3_" + str(k) +".txt"
    np.savetxt(kmeansFile, clusters, fmt="%s")

    print("SAVING KMEANS MODEL")
    joblib.dump(model, "bovw/results/KMEANS/models/model_KMEANS_TRAIN3_" + str(k) + ".pkl", compress=3)
    print("Done\n")


