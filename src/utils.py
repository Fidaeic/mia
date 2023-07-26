#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:11:09 2021

@author: fidae
"""
import numpy as np
from numpy import linalg as LA

from exceptions import CustomException
from rasterio.plot import show


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