'''kmeans.py
Performs K-Means clustering
Adrian Gellert
CS 251: Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors
from scipy import linalg as la

class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        print(data.shape)
        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data
        self.num_features, self.num_samps = self.data.shape[0], self.data.shape[1]

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        copy = self.data.copy()
        return copy

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        distance = np.sqrt(np.sum(np.square(pt_1 - pt_2),axis = 0))
        return distance

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        distances = np.sqrt(np.sum(np.square(pt - centroids), axis=1))
        return distances

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        rand_init_means = np.random.choice(self.num_samps, k)
        centroids = self.data[rand_init_means, :]
        self.centroids = centroids
        self.k = k
        return centroids

    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        # Initialize self.centroids and self.k
        self.centroids = self.initialize(k).astype(np.float16)
        random_value = np.random.randint(0, self.num_samps)
        pt = self.data[random_value]

        cur_iter = 0
        while cur_iter < max_iter:           
            # Compute distances between data points and centroids
            distances = self.dist_pt_to_centroids(pt, self.centroids)
            
            # Assign data points to nearest centroid
            self.data_centroid_labels = self.update_labels(self.centroids)

            # Update centroids
            # self.centroids, difference = self.update_centroids(self.k, self.data_centroid_labels, self.centroids)
            new_centroids, difference = self.update_centroids(self.k, self.data_centroid_labels, self.centroids)
            # self.inertia = self.compute_inertia()
            
            if abs(difference).any() < tol:
                break
            
            self.centroids = new_centroids
            cur_iter += 1
            
            # if abs(difference).any() > tol:
            #     cur_iter += 1
            #     print('continue iterating')
            # else:
            #     print('the break')
            #     break
        self.inertia = self.compute_inertia()
        
        if verbose == True:
            print("Total number of iterations: ", cur_iter)

        return self.inertia, cur_iter

    def cluster_batch(self, k=2, n_iter=1, verbose=False):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        """ lowest_inertia = float('inf')
        for i in range(n_iter):
            centroid = self.initialize(k)
            self.centroids = centroid
            inertia, _ = self.cluster(k, verbose=verbose)
            if inertia < lowest_inertia:
                lowest_inertia = inertia
                current_centroids = centroid
                current_labels = self.update_labels(self.centroids)
        self.centroids = current_centroids
        self.data_centroid_labels = current_labels
        self.inertia = lowest_inertia """
        centroids = []
        labels = []
        inertias = []
        
        # lowest_inertia = float('inf')
        for i in range(n_iter):
            centroid = self.initialize(k)
            centroids.append(centroid)
            
            labels.append(self.update_labels(centroid))
            
            # self.centroids = centroid
            inertia, iterations = self.cluster(k, verbose=verbose)
            inertias.append(inertia)
            
        best = np.argmin(inertias)
        
        self.centroids = centroids[best]
        self.data_centroid_labels = labels[best]
        self.inertia = inertias[best]
        
        #     if inertia < lowest_inertia:
        #         lowest_inertia = inertia
        #         current_centroids = centroid
        #         current_labels = self.update_labels(self.centroids)
        #         iterations = cur_iter
        # if lowest_inertia == float('inf'):
        #     # return float('inf'), None
        #     print('There was an error')
        # else:
        #     self.centroids = current_centroids
        #     self.data_centroid_labels = current_labels
        #     self.inertia = lowest_inertia
        #     return lowest_inertia, iterations

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        labels = np.zeros(self.num_samps, dtype=int)
        for (i,j), value in np.ndenumerate(self.data):
            distances = self.dist_pt_to_centroids(self.data[i], centroids)
            labels[i] = np.argmin(distances)
        self.data_centroid_labels = labels
        return labels

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        '''
        new_centroids = prev_centroids.copy()
        
        for i in range(k):
            cluster_data = self.data[data_centroid_labels == i,:]

            if cluster_data.shape[0] ==0:
                ran_idx = np.random.choice(self.num_samps, size=1, replace=False)
                new_centroids[i] = self.data[ran_idx, :]

            else:
                new_centroids[i] = np.mean(cluster_data, axis = 0)
        
        self.centroids = new_centroids
        return(new_centroids,new_centroids-prev_centroids)

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        inertia = 0
        for i in range(self.k):
            cluster_data = self.data[self.data_centroid_labels == i]
            distance = cluster_data - self.centroids[i]
            inertia += np.sum(distance**2)/self.data.shape[0]
        
        inertia = np.sum(inertia)
        return inertia

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''
        plt.scatter(self.data[:,0], self.data[:,1], c=self.data_centroid_labels, cmap='plasma')
        plt.scatter(self.centroids[:,0], self.centroids[:,1], c='black', marker='+')

    def elbow_plot(self, max_k, n_iter=1):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.
        n_iter: int. Number of iterations to run k-means for each k.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        x = np.arange(1,max_k+1)
        y = []
        for i in range(1,max_k+1):
            inertia,_ = self.cluster_batch(k=i, n_iter=n_iter)
            y.append(inertia)
            
        plt.plot(x,y)
        plt.xlabel('k clusters')
        plt.ylabel('inertia') 

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
    
        # # get a copy of the data
        # data_copy = self.get_data()
        # # get list of centroids and their labels
        # centroids = self.get_centroids()
        # centroid_labels = self.get_data_centroid_labels()
        # distances = []
        # # loop through each pixel in the image and replace it with the closest centroid value
        # for i in range(data_copy.shape[0]):
        #     # get the distance between the pixel and each centroid and find the minimum distance
        #     distances.append(self.dist_pt_to_centroids(data_copy[i,:], centroids))
        #     # get the index of the centroid with the minimum distance and replace the pixel with that centroid's label value
        #     data_copy[i] = centroid_labels[np.argmin(distances)]
        # # replace the data with the new data (RGB)
        # self.data = data_copy
        
        # Convert RGB values to floats in [0, 1] range
        # print(f"data: {self.data}")
        # self.data = self.data.astype(float) / 255.0
        
        # Get a copy of the data
        data = self.get_data()
        # Get data-to-cluster assignments
        labels = self.get_data_centroid_labels()
        # print(f"labels: {labels}")
        # Get centroids
        # print(f"centroids type: {type(self.get_centroids())}")
        # print(f"centroids data: {self.get_centroids()}")
        centroids = self.get_centroids()
        # Replace each RGB value with its assigned centroid
        for i in range(self.num_samps):
            data[i] = centroids[labels[i]]
        # assign the new data to the image
        self.data = data