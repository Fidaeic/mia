#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 19:23:12 2021

@author: fidae
"""

import rasterio as rio
from rasterio.plot import show
import numpy as np
import utils
from statsmodels.multivariate.pca import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class PCAMIA():
    def __init__(self, path_to_image):
        self.path_to_directory = path_to_image
        
       
    def read_image(self):
        
        self.image = rio.open(self.path_to_directory).read()
        self.rows = self.image.shape[1]
        self.columns = self.image.shape[2]
        
    def plot_image(self):
        
        show(self.image)
        
    def batchwise(self, neighbours):
        '''
        Generates a new matrix with the newighbouring pixels of every pixel of each band.
        The rows of the outcoming matrix represents pixels of the original image and the
        columns are the values of the pixel and the neighbouring ones, given a number of
        neighbours

        Parameters
        ----------
        neighbours : int
            Defines how many neighbours should be considered to build the matrix.

        Returns
        -------
            numpy ndarray with the bands and the neighbouring pixels extended batchwise
        '''

        depth = self.image.shape[0]
        
        final_matrix = np.array([])
        
        for d in range(depth):
            
            wind = utils.window(self.image[d, :, :], neighbours)
            
            if final_matrix.shape[0]==0:
                final_matrix = wind
            else:
                final_matrix = np.append(final_matrix, wind, axis=1)
        
        self.extended_image = final_matrix
        self.new_rows = self.rows-2*neighbours
        self.new_columns = self.columns-2*neighbours
    
    def reduce(self, ncomps):
        
        reduced = PCA(data=self.extended_image, ncomp=ncomps, standardize=True, demean=True, normalize=True, method='nipals')

        print(reduced.rsquare)
    
        self.scores, self.loadings = reduced.scores, reduced.loadings
#        
    def score_map(self, component):
    
        score_plot = np.reshape(self.scores[:, component], newshape=(self.new_rows, self.new_columns))
    
        plt.imshow(score_plot, interpolation='bilinear', cmap='RdBu')
        plt.colorbar()
        plt.title(f"Score map for component {component+1}")
    
        plt.show()
#        
#    def loadings_plot(self):
        
    def clusters(self, n_neighbours):
        knn = KMeans(n_neighbours)
    
        knn.fit(self.scores)
        
        self.clusters_labels = knn.labels_
        
        cluster_map= np.reshape(self.clusters_labels, newshape=(self.new_rows, self.new_columns))
    
        plt.imshow(cluster_map, interpolation='bilinear', cmap='RdBu')
        plt.colorbar()
        plt.title(f"Cluster map with n={n_neighbours}")
    
        plt.show()
#        
#    def scatter_scores(self):
#        
#    def save_image(self):
        
pca = PCAMIA('../data/Cubillas.jpg')

pca.read_image()

pca.plot_image()

pca.batchwise(2)

pca.extended_image.shape

pca.reduce(5)

pca.score_map(3)

pca.clusters(5)

