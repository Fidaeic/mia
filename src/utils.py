#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:11:09 2021

@author: fidae
"""
import numpy as np

def ndvi(b8, b4):
    return (b8-b4)/(b8+b4)

def gndvi(b8, b3):
    return (b8-b3)/(b3+b8)

def avi(b8, b4):
    return (b8*(1-b4)*(b8-b4))**(1/3)

def savi(b8, b4):
    return (b8-b4)/((b8+b4+0.428)*1.428)

def ndmi(b8, b11):
    return (b8-b11)/(b8+b11)

def msi(b8, b11):
    return b11/b8

def gci(b9, b3):
    return (b9/b3)-1

def nbri(b8, b12):
    return (b8-b12)/(b8+b12)

def bsi(b2, b4, b8, b11):
    return ((b11+b4)-(b8+b2))/((b11+b4)+(b8+b2))

def evi(b8, b4, b2):
    return 2.5*((b8-b4)/(b8+6*b4-7.5*b2+1))

def ndwi(b8, b3):
    return (b3-b8)/(b3+b8)


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
