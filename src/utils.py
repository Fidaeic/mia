#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:11:09 2021

@author: fidae
"""
import numpy as np
from numpy import linalg as LA

def window(X, neighbours):
    '''
    Gets the neighbouring pixels of every pixel of the image. The window size determines
    how big is the neighbourhood of the pixel, and its given by the following expression:
            window_size = (2*n+1)**2
    being n the number of neighbours

    Parameters
    ----------
    neighbours : int
        Integer defining how many neighbours should be considered to build the matrix

    Returns
    -------
    new_matrix : numpy ndarray
        2d matrix containing the neighbouring pixels .

    '''
    # Get the number of rows and columns of the new image. We must subtract
    # as many neighbours as introduced as a parameter
    nrows = X.shape[0]-2*neighbours
    ncols = X.shape[1]-2*neighbours


    window_size = (2*neighbours + 1)**2

    new_matrix = np.empty(shape=(nrows*ncols, window_size))
    p = 0
    for i in range(neighbours, X.shape[0]-neighbours):
        for j in range(neighbours, X.shape[1]-neighbours):

            mask = X[i-neighbours: i+neighbours+1, j-neighbours:j+neighbours+1]

            vector = np.reshape(mask, newshape=mask.shape[0]*mask.shape[1])

            new_matrix[p, :] = vector

            p+=1

    return new_matrix
