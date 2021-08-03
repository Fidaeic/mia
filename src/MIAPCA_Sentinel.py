#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 19:23:12 2021

@author: fidae
"""

import rasterio as rio
import numpy as np
from utils import *

class PCAMIA():
    def __init__(self, path_to_directory):
        
        
        self.paths = {"b2" : path_to_directory+'T30SVG_20201030T110211_B02_60m.jp2',
        "b3" : path_to_directory+'T30SVG_20201030T110211_B03_60m.jp2',
        "b4" : path_to_directory+'T30SVG_20201030T110211_B04_60m.jp2',
        "b8" : path_to_directory+'T30SVG_20201030T110211_B8A_60m.jp2',
        "b9" : path_to_directory+'T30SVG_20201030T110211_B09_60m.jp2',
        "b11" : path_to_directory+'T30SVG_20201030T110211_B11_60m.jp2',
        "b12" : path_to_directory+'T30SVG_20201030T110211_B12_60m.jp2',
        "rgb" : path_to_directory+'T30SVG_20201030T110211_TCI_60m.jp2'}
        
       
    def get_indices(self):
        
        b2 = rio.open(paths_dict['b2']).read(1)[:extension, :extension]
        b3 = rio.open(paths_dict['b3']).read(1)[:extension, :extension]
        b4 = rio.open(paths_dict['b4']).read(1)[:extension, :extension]
        b8 = rio.open(paths_dict['b8']).read(1)[:extension, :extension]
        b9 = rio.open(paths_dict['b9']).read(1)[:extension, :extension]
        b11 = rio.open(paths_dict['b11']).read(1)[:extension, :extension]
        b12 = rio.open(paths_dict['b12']).read(1)[:extension, :extension]
        rgb = rio.open(paths_dict['rgb']).read()[:, :extension, :extension]
    
        ndvi_ar = ndvi(b8, b4)
        gndvi_ar = gndvi(b8, b3)
        avi_ar = avi(b8, b4)
        savi_ar = savi(b8, b4)
        ndmi_ar = ndmi(b8, b11)
        msi_ar = msi(b8, b11)
        gci_ar = gci(b9, b3)
        nbri_ar = nbri(b8, b12)
        bsi_ar = bsi(b2, b4, b8, b11)
        evi_ar = evi(b8, b4, b2)
        ndwi_ar = ndwi(b8, b3)
    
        self.indices_image = np.stack([b2, b3, b4, b8, b9, b11, b12, ndvi_ar, gndvi_ar, avi_ar, \
                      savi_ar, ndmi_ar, msi_ar, gci_ar, nbri_ar, bsi_ar, evi_ar, ndwi_ar, rgb[0], rgb[1], rgb[2]], axis=2)
        

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

        depth = self.indices_image.shape[2]
        
        final_matrix = np.array([])
        
        for d in range(depth):
            
            wind = window(self.indices_image[:, :, d], neighbours)
            
            if final_matrix.shape[0]==0:
                final_matrix = wind
            else:
                final_matrix = np.append(final_matrix, wind, axis=1)
        
        self.extended_image = final_matrix
    
    def reduce(self):
        
    def score_map(self):
        
    def loadings_plot(self):
        
    def scatter_scores(self):
        
    def save_image(self):
        
        