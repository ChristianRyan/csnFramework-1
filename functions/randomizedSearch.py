#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 13:14:27 2018

@author: slim
"""
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
import math
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier


def randomSearch(dataset,n_iter,classifier=KNeighborsClassifier(n_neighbors=1), number_of_values = 100,scores=['accuracy','f1_macro', 'f1_micro']):
    returnmainDf = pd.DataFrame(columns = ['Strategy', 'Dataset', 'n_instances', 'l_attributes', 'k_neighbours', 'fit_time', 'accuracy', 'f1_macro', 'f1_micro'])

    returnListDfs = []

    for idx, data in enumerate(dataset):
        returnDf = pd.DataFrame(columns = ['Strategy', 'Dataset', 'n_instances', 'l_attributes', 'k_neighbours', 'fit_time', 'accuracy', 'f1_macro', 'f1_micro'])
        print("New dataset",data.shape[0])
        n = data.shape[0]

        gridparams = {'n_neighbors' : np.linspace(1, round(math.sqrt(n)), number_of_values).astype(int)}
        clf = RandomizedSearchCV(classifier, gridparams, n_iter=n_iter,n_jobs=-1,scoring=scores,refit=False)
        y = data['target']
        X = data.drop('target', axis=1)
        clf.fit(X, y)
        resultsGridSearch = pd.DataFrame(clf.cv_results_)
        #print(resultsGridSearch)
        returnDf[['fit_time', 'accuracy', 'f1_macro', 'f1_micro']] = resultsGridSearch[['mean_fit_time', 'mean_test_accuracy', 'mean_test_f1_macro', 'mean_test_f1_micro']]
        returnDf['Strategy'] = 'Randomised search'
        returnDf['Dataset'] = 'D' + str(idx)
        returnDf['n_instances'] = n
        returnDf['l_attributes'] = len(data.columns)
        returnDf['k_neighbours'] = resultsGridSearch['param_n_neighbors']
        returnListDfs.append(returnDf)


    for dataf in returnListDfs:
        returnmainDf = returnmainDf.append(dataf)

    return returnmainDf
