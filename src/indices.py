#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:11:09 2021

@author: fidae
"""

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