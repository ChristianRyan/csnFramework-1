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
    returnmainDf = pd.DataFrame(columns = ['Strategy', 'Dataset', 'n_instances', 'l_attributes', 'k_neighbours', 'fit_time', 'accuracy', 'recall_macro', 'recall_micro', 'precision_macro', 'precision_micro'])
    
    returnListDfs = []
    returnDf = pd.DataFrame(columns = ['Strategy', 'Dataset', 'n_instances', 'l_attributes', 'k_neighbours', 'fit_time', 'accuracy', 'recall_macro', 'recall_micro', 'precision_macro', 'precision_micro'])
    for idx, data in enumerate(dataset):
        print("New dataset")
        n = data.size
        
        gridparams = {'n_neighbors' : np.linspace(1, round(math.sqrt(n)), number_of_values).astype(int)}
        clf = RandomizedSearchCV(classifier, gridparams, n_iter=n_iter,n_jobs=-1,scoring=scores,refit=False) # TODO: BE careful with cores :D
        y = data['target']
        X = data.drop('target', axis=1)
        clf.fit(X, y)
        resultsGridSearch = pd.DataFrame(clf.cv_results_)
        returnDf[['fit_time', 'accuracy', 'recall_macro', 'recall_micro', 'precision_macro', 'precision_micro']] = resultsGridSearch[['mean_fit_time', 'mean_test_accuracy', 'mean_test_recall_macro', 'mean_test_recall_micro', 'mean_test_precision_macro', 'mean_test_precision_micro']]
        returnDf['Strategy'] = 'Brute Force'
        returnDf['Dataset'] = 'D' + str(idx)
        returnDf['n_instances'] = n
        returnDf['l_attributes'] = len(data.columns)
        returnDf['k_neighbours'] = 'D' + str(idx) # TODO: Fix k_neighbours
        returnListDfs.append(returnDf)

        #returnDict['dataset '+idx] = ''
        
    for dataf in returnListDfs:
        returnmainDf = returnmainDf.append(dataf)

    return returnmainDf