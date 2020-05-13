
import numpy as np
from scipy.spatial import distance
from scipy.cluster.vq import vq
from define_class import defineClass, defineClassWithArtists, defineClassWithOnlyFilename
from utils import *
import time 

# takes two arrays as parameters and find the l1 distance
def L1_dist(vec1, vec2):
    return np.linalg.norm(np.subtract(np.array(vec1), np.array(vec2)))  

# Find the index of the closest central point to the each sift descriptor. 
# Takes 2 parameters the first one is a sift descriptor and the second one is the array of central points in k means
# Returns the index of the closest central point.  
def find_index(image, center):
    count = 0
    ind = 0
    
    for i in range(len(center)):
        if(i == 0):
           count = distance.euclidean(image, center[i]) 
           #count = L1_dist(image, center[i])
           #print(count)
        else:
            dist = distance.euclidean(image, center[i]) 
            #dist = L1_dist(image, center[i])
            if(dist < count):
                ind = i
                count = dist
    return ind

# Takes 2 parameters. The first one is a dictionary that holds the descriptors that are separated class by class 
# And the second parameter is an array that holds the central points (visual words) of the k means clustering
# Returns a dictionary that holds the histograms for each images that are separated class by class. 
def image_class(all_bovw, centers):
    dict_feature = {}
    for image_name,descriptors in all_bovw.items():
        category = []
        for img in descriptors:
            histogram = np.zeros(len(centers))
            
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[image_name] = category
    return dict_feature


#with the k-means model found, this code generates the feature vectors 
#by building an histogram of classified keypoints in the kmeans classifier 
def calculate_centroids_histogram(images,descriptors, model, k=100):

    feature_vectors=[]
    class_vectors=[]
    
    # Console message
    status = ""

    # faire histogramme de chaque image et l'ajouter
    for image in images:
        # Start timer
        start_timeHIST = time.time()

        des = descriptors[image]
        #classification of all descriptors in the model
        predict_kmeans=model.predict(des)
        
        #calculates the histogram
        hist, bin_edges=np.histogram(predict_kmeans, bins=k)

        #histogram is the feature vector
        feature_vectors.append(hist)

        end_timeHIST = time.time()
        elapsed_timeHIST = end_timeHIST - start_timeHIST
        status = "COMPUTING HIST :" + str(elapsed_timeHIST)[0:4]

        #status
        printProgressBar (len(feature_vectors), len(images), prefix = image, suffix = status)
        
    
    #TRAIN 1 & 2
    #class_vectors = defineClass(images)

    # TRAIN 3
    #class_vectors = defineClassWithArtists(images)

    # TRAIN 4
    class_vectors = defineClassWithOnlyFilename(images)

    feature_vectors=np.asarray(feature_vectors)
    #class_vectors=np.asarray(class_vectors)
    #return vectors and classes we want to classify
    return class_vectors, feature_vectors

#with the k-means model found, this code generates the feature vectors 
#by building an histogram of classified keypoints in the kmeans classifier 
def calculate_centroids_histogram_v2(images,descriptors, model, k=100):

    feature_vectors=[]
    class_vectors=[]
    

    # faire histogramme de chaque image et l'ajouter
    for image in images:

        des = descriptors[image]

        histogram = np.zeros(len(model.cluster_centers_))
        
        # Prediction
        cluster_result =  model.predict(des)

        for i in cluster_result:
            histogram[i] += 1.0
        

        #histogram is the feature vector
        feature_vectors.append(histogram)

        #status
        printProgressBar (len(feature_vectors), len(images), prefix = image)

    # TRAIN 4
    class_vectors = defineClassWithOnlyFilename(images)

    feature_vectors=np.asarray(feature_vectors)
    #class_vectors=np.asarray(class_vectors)
    #return vectors and classes we want to classify
    return class_vectors, feature_vectors


#with the k-means model found, this code generates the feature vectors 
#by building an histogram of classified keypoints in the kmeans classifier 
def calculate_centroids_histogram_v3(images,descriptors, model, k=100):

    feature_vectors=[]
    class_vectors=[]
    
    status = ""
    # faire histogramme de chaque image et l'ajouter
    for image in images:
        

        des = descriptors[image]
        
        # Prediction
        cluster_result =  model.predict(des)

        hist = vq(des, model.cluster_centers_)
        
        #histogram is the feature vector
        feature_vectors.append(hist)

        #status
        printProgressBar(len(feature_vectors), len(images), prefix = image)

    # TRAIN 4
    class_vectors = defineClassWithOnlyFilename(images)

    feature_vectors=np.asarray(feature_vectors)
    #class_vectors=np.asarray(class_vectors)
    #return vectors and classes we want to classify
    return class_vectors, feature_vectors