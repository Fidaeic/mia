#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:11:09 2021

@author: fidae
"""
import os
from exceptions import CustomException
import sys
from rasterio.plot import show
import math
import matplotlib.pyplot as plt
import numpy as np
import dill
from components.data_ingestion import DataIngestion


def plot_image(image):
        '''
        Plots original image

        Returns
        -------
        RGB image.

        '''
        
        try:
            show(image)

        except Exception as e:
            raise CustomException(e, sys)
        
def score_map(scores):
        '''
        2D maps that represent the scores of the principal components. When compared,
        score maps of different principal components are very useful for feature extraction

        Returns
        -------
        Matplotlib plot with the score map of the specified component.

        '''
        try:

            _, rows, columns = DataIngestion()

            # Gets the number of required columns for the subplot
            n_columns = math.ceil(scores.shape[1]/2)
            
            # Generation of subplots based on the number of columns
            fig, axs = plt.subplots(n_columns, 2, figsize=(20,15))
            
            # Check if the number of components is an odd number. If so, delete
            # the last image
            if scores.shape[1] % 2 != 0:
                fig.delaxes(axs[-1,-1])
            
            fig.subplots_adjust(hspace = .5, wspace=.1)
            axs = axs.ravel()
            
            # Generate a score map for each component 
            for i in range(scores.shape[1]):
                score_plot = np.reshape(scores[:, i], newshape=(rows-2, columns-2))
            
                im = axs[i].imshow(score_plot, interpolation='bilinear', cmap='RdBu')
                axs[i].set_title(f"Component {i+1}", fontsize=14)
            
                fig.colorbar(im, ax=axs[i])
                plt.tight_layout()

        except Exception as e:
            raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:

        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    