#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:11:09 2021

@author: fidae
"""
import numpy as np
from numpy import linalg as LA

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

def nipals(X, ncomps, threshold=1e-5, demean=True, standardize=True, verbose=True, max_iterations=10000, simplified=True):
  
    X_pca = X.copy()

    if demean:
        mean = np.mean(X, axis=0)
        X_pca = X_pca-mean[None, :]
        
    if standardize:
        std = np.std(X, axis=0)
        X_pca = X_pca/std[None, :]


    tss = np.sum(X_pca**2)
    
    r2 = []
    explained_variance = []
    T = np.zeros(shape=(ncomps, X_pca.shape[0]))
    P_t = np.zeros(shape = (ncomps, X_pca.shape[1]))
    eigenvalues= np.zeros(ncomps)

    for i in range(ncomps):
        # We get the column with the maximum variance in the matrix
        var = np.var(X_pca, axis=0)
        pos = np.where(max(var))[0]
        
        # That column will be the one we will start with
        t = np.array(X_pca[:,pos])
        t.shape=(X_pca.shape[0], 1) #Esto sirve para obligar a que t sea un vector columna
    
        cont=0
        comprobacion = 1
        # while conv <X_pca.shape[0] and cont<10000:
        while comprobacion>threshold and cont<max_iterations:
            
            #Definimos un vector llamado t_previo, que es con el que vamos a empezar el algoritmo
            t_previo = t
            p_t = (np.transpose(t_previo).dot(X_pca))/(np.transpose(t_previo).dot(t_previo))
            p_t = p_t/LA.norm(p_t)
            
            t=X_pca.dot(np.transpose(p_t))
            
            #Comparamos el t calcular con el t_previo, de manera que lo que buscamos es que la diferencia sea menor
            #que el criterio de parada establecido. Para ello, hacemos una prueba lógica y sumamos todos los valores
            #donde sea verdad. Si es verdad en todos, el algoritmo ha convergido
            
            t_sum = np.sum(t**2)
            t_previo_sum = np.sum(t_previo**2)
            
            comprobacion = np.abs(np.sqrt(t_sum-t_previo_sum))
            
            cont+=1

        #Calculamos la matriz de residuos y se la asignamos a X para calcular la siguiente componente
        E = X_pca-t.dot(p_t)
        r2.append(1-np.sum(E**2)/tss)
        explained_variance.append(r2[i] - r2[i-1]) if i!=0 else explained_variance.append(r2[i])
        X_pca = E
        
        #Asignamos los vectores t y p a su posición en las matrices de scores y loadings
        eigenvalues[i] = np.var(t)
        
        T[i]=t.reshape((X.shape[0]))
        P_t[i]=p_t
        
    if verbose:
        print(f"Algorithm converged in {cont} iterations")
    T = np.transpose(T)

    if simplified:
        return T, P_t, r2

    return T, P_t, E, r2, explained_variance, eigenvalues
