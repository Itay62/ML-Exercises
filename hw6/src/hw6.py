import numpy as np

def get_random_centroids(X, k):

    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###############################################     ############################
    #for i in range(k):
    
    r =(tuple(np.random.randint(np.min(X[:,0]),np.max(X[:,0]),k))) #k random values for the r color  in the data range
    g = (tuple(np.random.randint(np.min(X[:,1]),np.max(X[:,1]),k))) #k random values for the g color in the data range
    b = (tuple(np.random.randint(np.min(X[:,2]),np.max(X[:,2]),k))) #k random values for the b color in the data range
    centroids = list(zip(r, g,b))


    ########################
    # ###################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float)



def lp_distance(X, centroids, p=2):

    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    k = len(centroids)
    distances = [] 
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    #for each centroid check the minkowski_distance for all instances in X
    for i in range(k): 
        distances.append(minkowski_distance(X,centroids[i],p))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return np.stack(distances)


def minkowski_distance(x, y, p_value):
    '''
    Inputs:
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of distances between all points to a specific centroid
    '''
    power = np.abs(np.subtract(x,y))
    power = power ** p_value
    #for each instance sum the rgb values and append it to the distancesForK list
    distance = np.sum(power,1)
    return distance ** (1/p_value)    
 
# def minkowski_distance(x, y, p_value):
#     '''
#     Inputs:
#     A single image of shape (num_pixels, 3)
#     The centroids (k, 3)
#     The distance parameter p

#     output: numpy array of distances between all points to a specific centroid
#     '''
#     distancesForK = []
#     power = np.power(np.abs(np.subtract(x,y)), p_value)
#     #for each instance sum the rgb values and append it to the distancesForK list
#     for instance in power:
#         distance = np.sum(instance)
#         distancesForK.append(distance) 
#     return np.power(distancesForK,1/p_value)


def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    last_centroids = np.copy(centroids)
    #   ##########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for i in range(max_iter):

        distances = lp_distance(X,centroids,p) 
        classes = np.argmin(distances, axis=0) ## an array of closest centroid to an instance
        

        for c in set(classes):
            cluster_intances = X[classes == c] ## all instances that which belongs to this(c) centroid
            centroids[c,:] = np.mean(cluster_intances,axis = 0) ## new centroid = mean of the cluster_instances  
        if ((last_centroids == centroids).all()):
            break
        
        last_centroids = np.copy(centroids)
  
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes
# def kmeans(X, k, p ,max_iter=100):
#     """
#     Inputs:
#     - X: a single image of shape (num_pixels, 3).
#     - k: number of centroids.
#     - p: the parameter governing the distance measure.
#     - max_iter: the maximum number of iterations to perform.

#     Outputs:
#     - The calculated centroids as a numpy array.
#     - The final assignment of all RGB points to the closest centroids as a numpy array.
#     """
#     centroids = get_random_centroids(X, k)
#     iterations = 0

#     #   ##########################################################################
#     # TODO: Implement the function.                                           #
#     ###########################################################################
#     while (iterations <= max_iter):
#         new_centroids = []
#         distances = lp_distance(X,centroids,p)
#         classes = np.argmin(distances, axis=0)
#         for i in range(k):
#             centroid_instances = X[np.where(classes == i)]
#             centroid = np.around(np.mean(centroid_instances,0))
#             new_centroids.append(centroid)
#         new_centroids = np.stack(new_centroids)        
#         if (new_centroids == centroids).all():
#             break
#         else:
#             centroids = new_centroids
#             iterations += 1
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     return centroids, classes

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = []
    
    index = np.random.randint(X.shape[0])
    first_centroid = X[index].astype(np.float)
    centroids.append(first_centroid)
    X_copy = np.copy(X)
    X_copy = np.delete(X_copy,index,0)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for i in range(k-1):
        distance = lp_distance(X_copy, centroids, p=2)
        if distance.shape[0] > 1:
            distance = np.amin(distance,axis=0)
        sum = np.sum(distance)
        distance = np.true_divide(distance, sum)
        cent_index = np.argmax(distance)
        centroids.append(X_copy[cent_index].astype(np.float))
        X_copy = np.delete(X_copy,cent_index,0) 
        
    centroids = np.stack(centroids)
    last_centroids = np.copy(centroids)


    for i in range(max_iter):
        distances = lp_distance(X,centroids,p)
        classes = np.argmin(distances, axis=0)
        for c in set(classes):
            cluster_intances = X[classes == c]
            centroids[c,:] = np.mean(cluster_intances,axis = 0)
        if ((last_centroids == centroids).all()):
            break
        
        last_centroids = np.copy(centroids)
  
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes

def inertia_for_k_clusters (X, centroids, classes, p):
    inertia = 0
    for c in set(classes):
        cluster_instances = X[classes == c]
        instance_dist_cluster = lp_distance(cluster_instances,[centroids[c]],p)
        inertia += np.sum(instance_dist_cluster)
    return inertia  