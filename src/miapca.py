#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 19:23:12 2021

@author: fidae el morer
"""

import rasterio as rio
from rasterio.plot import show
import numpy as np
import utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import math

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from exceptions import CustomException
import sys
import pickle as pk


class MIAPCA():
    def __init__(self, path_to_image):
        '''
        This class is intended for analyzing multiband images through color-texture analysis.
        It extends a multiband image batchwise and applies a PCA for dimensionality reduction
        and feature extraction

        Parameters
        ----------
        path_to_image : string
            Path to the image to be analyzed.

        '''
        self.path_to_directory = path_to_image

        self.scaler = StandardScaler()       
        
       
    def read_image(self):
        '''
        Reads image from path and stores it as an attribute. The method also 
        takes the number of rows and columns as attributes

        Returns
        -------
        Original image as numpy ndarray
        Number of rows and columns.

        '''

        try:
        
            self.image = rio.open(self.path_to_directory).read()
            self.rows = self.image.shape[1]
            self.columns = self.image.shape[2]

        except Exception as e:
            raise CustomException(e, sys)
        
    def plot_image(self):
        '''
        Plots original image

        Returns
        -------
        RGB image.

        '''
        
        try:
            show(self.image)

        except Exception as e:
            raise CustomException(e, sys)
        
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

        self.neighbours = neighbours

        try:
            # Get the number of bands of the image
            depth = self.image.shape[0]
            
            # Create an empty array that will contain the original image extended batchwise
            
            final_matrix = np.array([])
            
            for d in range(depth):
                
                # Apply the window function to each band of the original image
                wind = utils.window(self.image[d, :, :], neighbours)
                
                # Append the extended band to the final matrix
                if final_matrix.shape[0]==0:
                    final_matrix = wind
                else:
                    final_matrix = np.append(final_matrix, wind, axis=1)
            
            self.extended_image = final_matrix
            self.new_rows = self.rows-2*neighbours
            self.new_columns = self.columns-2*neighbours

        except Exception as e:
            raise CustomException(e, sys)
    
    def fit(self, ncomps):
        '''
        Applies PCA to the image after extending it batchwise

        Parameters
        ----------
        ncomps : int
            Number of components to be considered for the PCA.

        Returns
        -------
        Scores: numpy ndarray
            Coordinates of the extended image after projecting it on a reduced
            subspace
        Loadings: numpy ndarray
            Set of vectors that define the directions of the principal components
            in the new subspace
        Rsquare: list
            Accumulated explained variance for each component

        '''

        if self.extended_image==None:
            raise CustomException("No batchwise image has been generated. Please apply the batchwise() method", sys)

        try:

            self.ncomps = ncomps

            self.X_scaled = self.scaler.fit_transform(self.extended_image)

            self.pca_object = PCA(n_components=self.ncomps)

            self.scores = self.pca_object.fit_transform(self._X_scaled)

            self.loadings = self.pca_object.components_

            self.rsquare = np.cumsum(self.pca_object.explained_variance_ratio_)

            self.explained_variance = self.pca_object.explained_variance_ratio_

            self.eigenvals = self.pca_object.singular_values_**2

            self.component_variance = np.var(self._scores, axis=0)

            self.reconstructed_training_X, self.residuals, _ = self.reconstruct(self.X)

            # Save the PCA object after training
            pk.dump(self.pca_object, open("../artifacts/pca.pkl", "wb"))

        except Exception as e:

            raise CustomException(e, sys)
        
    def project(self, X_predict):
        '''
        Projection of new samples onto the subspace created by the principal components

        Parameters
        ----------
        
        X_predict: ndarray
            New samples to be projected

        Returns
        -------
        predicted_scores: ndarray
            Scores or coordinates of the samples on the new subspace
        '''
        try:
            X_predict_scaled = self.scaler.transform(X_predict)
            
            return self.pca_object.transform(X_predict_scaled)
        
        except Exception as e:
            raise CustomException(e, sys)


    def reconstruct(self, X_predict):
        '''
        Reconstruction of a data set after projecting it to the lower dimensional space. The data set is first
        standardized and projected onto the latent space, and afterwards it is reprojected onto the original space.
        To compute the errors, we have to scale the original data
        '''

        try:

            # Get the scores of the set
            predicted_scores = self.project(X_predict)

            # Reconstruct the original data set based on the scores
            X_hat = self.pca_object.inverse_transform(predicted_scores)

            # Rescale the prediction data set
            X_predict_rescaled = self.scaler.transform(X_predict)

            # Compute the difference between the predicted and the true data E = X-TP^t
            errors = X_predict_rescaled - X_hat

            tss_test = np.sum((X_predict-self.scaler.mean_)**2, axis=0)
            rss_test = np.sum((X_predict-self.scaler.inverse_transform(X_hat))**2, axis=0)

            q2 = np.average(1- rss_test/tss_test)
            
            return X_hat, errors, q2
        
        except Exception as e:
            raise CustomException(e, sys)

    def score_map(self):
        '''
        2D maps that represent the scores of the principal components. When compared,
        score maps of different principal components are very useful for feature extraction

        Returns
        -------
        Matplotlib plot with the score map of the specified component.

        '''
        try:
            # Gets the number of required columns for the subplot
            n_columns = math.ceil(self.scores.shape[1]/2)
            
            # Generation of subplots based on the number of columns
            fig, axs = plt.subplots(n_columns, 2, figsize=(20,15))
            
            # Check if the number of components is an odd number. If so, delete
            # the last image
            if self.scores.shape[1] % 2 != 0:
                fig.delaxes(axs[-1,-1])
            
            fig.subplots_adjust(hspace = .5, wspace=.1)
            axs = axs.ravel()
            
            # Generate a score map for each component 
            for i in range(self.scores.shape[1]):
                score_plot = np.reshape(self.scores[:, i], newshape=(self.new_rows, self.new_columns))
            
                im = axs[i].imshow(score_plot, interpolation='bilinear', cmap='RdBu')
                axs[i].set_title(f"Component {i+1}", fontsize=14)
            
                fig.colorbar(im, ax=axs[i])

        except Exception as e:
            raise CustomException(e, sys)
       
    def loadings_plot(self):
        '''
        barplots that represent the loadings of the principal components. When compared,
        score maps of different principal components are very useful for feature extraction

        Returns
        -------
        Matplotlib plot with the bar plots of the principal components.

        '''
        try: 
            # Gets the number of required columns for the subplot
            n_rows = math.ceil(self.loadings.shape[0]/2)

            # Generation of subplots based on the number of columns
            fig, axs = plt.subplots(n_rows, 2, figsize=(20,15))

            # Check if the number of components is an odd number. If so, delete
            # the last image
            if self.loadings.shape[0] % 2 != 0:
                fig.delaxes(axs[-1,-1])

            fig.subplots_adjust(hspace = .5, wspace=.1)
            axs = axs.ravel()

            window_size = (2*self.neighbours + 1)**2
            vertical_coords = [coord-0.5 for coord in range(1, self.loadings.shape[1]) if coord % window_size==0]

            # Generate a score map for each component 
            for i in range(self.loadings.shape[0]):

                component_loadings = self.loadings[i, :]
                y_min = np.minimum(0, np.min(component_loadings))
                y_max = np.max(component_loadings)

                axs[i].bar(x=list(range(self.loadings.shape[1])), height=component_loadings)

                axs[i].vlines(vertical_coords, ymin=y_min, ymax=y_max, linestyle='--', color='red')

                axs[i].set_title(f"Loadings for component {i}")

        except Exception as e:
            raise CustomException(e, sys)

    def show_mask(self, mask, alpha=.5):

        try:

            score_plot = np.reshape(mask, newshape=(self.new_rows, self.new_columns))

            # fig, ax =  plt.subplots(figsize=(20, 10))

            image = mpimg.imread(self.path_to_directory)
            plt.imshow(image)
            plt.imshow(score_plot, cmap=plt.get_cmap('binary'), alpha=alpha)
            
            # fig.colorbar(im, ax=ax)
            plt.title("Dummy image for the selected features")

            plt.show()

        except Exception as e:
            raise CustomException(e, sys)

        
    def clusters(self, n_clusters):
        '''
        Applies a clustering method to the scores matrix after reducing the extended
        image. Currently only supports KMeans.

        Parameters
        ----------
        n_clusters : int
            Number of clusters to generate.

        Returns
        -------
        Matplotlib plot with a cluster map.

        '''
        try:
        
            # Create the cluster object with the specified number of clusters
            # and fit it to the score matrix
            knn = KMeans(n_clusters)
        
            knn.fit(self.scores)
            
        
            self.clusters_labels = knn.labels_
            
            # Create a 2d array where every pixel is represented with a cluster number        
            cluster_map= np.reshape(self.clusters_labels, newshape=(self.new_rows, self.new_columns))
            
            cmap = plt.get_cmap("RdBu", n_clusters)
        
            plt.imshow(cluster_map, interpolation='bilinear', cmap=cmap)
            plt.colorbar()
            plt.title(f"Cluster map with n={n_clusters}")
        
            plt.show()
        except Exception as e:
            raise CustomException(e, sys)
        
    def scatter_scores(self, component_1, component_2):
        '''
        Creates a scatter plot to compare the score of two components. May be useful
        to detect directions of variability and clusters

        Parameters
        ----------
        component_1, component_2 : int
            Principal components to display.

        Returns
        -------
        Matplotlib plot with a scatter plot comparing the scores of 2 principal components.

        '''
        try:
    
            plt.scatter(self.scores[:, component_1], self.scores[:, component_2])
            
            plt.axhline(color='red')
            plt.axvline(color='red')
            
            plt.title(f"Score plot for components {component_1+1} and {component_2+1}")
            
            plt.show()
        except Exception as e:
            raise CustomException(e, sys)

