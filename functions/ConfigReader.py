#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:54:01 2018

@author: slim
"""

import configparser

def ReadConfig():
    """  """
    config = configparser.ConfigParser()
    config.read('./config')
    params=config['params']
    k=params.get('k')
    interval=params.get('interval')
    searchAlgo=params.get('searchAlgo')
    optimization_Type=params.get('optimization_Type')
    typeOfClassifier=params.get('typeOfClassifier')
    radius=params.get('radius')
    weight=params.get('weight')
    if searchAlgo=='default':
        searchAlgo=["gridSearch"]
    if optimization_Type=='default':
        optimization_Type=['auto']
    if typeOfClassifier=='default':
        typeOfClassifier=['knn']
    if 'radiusN' in typeOfClassifier:
        if radius=='default':
            radius=1.0
        if weight=='default':
            weight=['uniform']
    return [k,interval,searchAlgo,optimization_Type,typeOfClassifier,radius,weight]



#%%