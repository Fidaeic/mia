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

def norm(x):
    return np.sqrt(np.sum(x * x))

def nipals(X, ncomps, threshold=1e-4, demean=True, standardize=True, verbose=True, max_iterations=1000000, simplified=True):

    if demean:
        mean = np.mean(X, axis=0)
        X = X-mean[None, :]

    if standardize:
        std = np.std(X, axis=0)
        X = X/std[None, :]

    tss = np.sum(X**2)

    cont=0
    for i in range(ncomps):
        # We start with the column that has the highest variance of the dataset. Therefore, we first check
        # which column has the highest variance
        var = np.var(X, axis=0)
        pos = np.argmax(var)

        # We select said column and assign it to t
        t = np.array(X[:,pos])
        t.shape=(X.shape[0], 1) #Esto sirve para obligar a que t sea un vector columna

        comprobacion = 1

        while comprobacion>threshold and cont<max_iterations:

            # Save a copy of t and compute the loadings vector for the component
            t_previo = t.copy()
            p_t = (np.transpose(t_previo).dot(X))/(np.transpose(t_previo).dot(t_previo))

            # Normalize the loadings vector of the component
            p_t = p_t/LA.norm(p_t)

            # Recompute the vector of scores t
            t = X.dot(np.transpose(p_t))

            # Convergence criterion: Difference of sum of squares smaller than threshold
            comprobacion = norm(t-t_previo)/norm(t)

            cont+=1

        # Computation of the error and the residual sum of squares       
        X -= t.dot(p_t)
        rss = np.sum(X**2) #Residual Sum of Squares

        # X is now the error matrix
        if i==0:
            T = t
            P_t = p_t
            r2 = [1-rss/tss]
            explained_variance = r2.copy()
        else:
            T = np.append(T, t, axis=1)
            P_t = np.append(P_t, p_t, axis=0)
            r2.append(1-rss/tss)
            explained_variance.append(r2[i] - r2[i-1])

        if verbose:
            print(f"Comp {i} converged in {cont} iterations")
    
    eigenvalues = np.var(T, axis=0)

    if simplified:
        return T, P_t, r2

    return T, P_t, X, r2, explained_variance, eigenvalues
