from sklearn.cluster import KMeans,MiniBatchKMeans

# A k-means clustering algorithm who takes 2 parameter which is number 
# of cluster(k) and the other is descriptors list(unordered 1d array)
# Returns an array that holds central points.
def kmeans(k, des):
    kmeans = MiniBatchKMeans(n_clusters = k,init='k-means++')
    kmeans.fit(des)
    y_kmeans = kmeans.predict(des)
    visual_words = kmeans.cluster_centers_ 
    return y_kmeans, visual_words, kmeans