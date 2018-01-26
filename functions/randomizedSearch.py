#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 13:14:27 2018

@author: slim
"""
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import math
from sklearn.metrics import *

def randomSearch(dataset,n_iter,classifier=KNeighborsClassifier(n_neighbors=1), number_of_values = 100,scores=['accuracy','recall_macro','recall_micro','precision_macro','precision_micro']):

    
    returnDf = pd.DataFrame(columns = ['Strategy', 'Dataset', 'n_instances', 'l_attributes', 'k_neighbours', 'fit_time', 'accuracy', 'recall_macro', 'recall_micro', 'precision_macro', 'precision_micro'])
    for idx, data in enumerate(dataset):
        print("New dataset")
        n = data.size
        
        gridparams = {'n_neighbors' : np.linspace(1, round(math.sqrt(n)), number_of_values).astype(int)}
        clf = RandomizedSearchCV(classifier, gridparams, n_iter=n_iter,n_jobs=-1,scoring=scores,refit=False) # TODO: BE careful with cores :D
        y = data['target']
        X = data.drop('target', axis=1)
        clf.fit(X, y)
        #for i in range(0,n_iter):
#            returnDf.append('RandomSearch','Dataset '+idx,n,)

        #returnDict['dataset '+idx] = ''
        
    # For now returns a list of dataframes that contain grid search result parameters (or rather should didnt test), should return optimal k
    return returnDf