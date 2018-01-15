# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:35:43 2018

@author: Christian
"""
import numpy as np
import pandas as pd
import math as math
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

def bruteForce(dataset, classifier, origin = 'default', interval = 'default'):
    n = int(dataset.describe().iloc[0,0])
    if isinstance(origin, str):
        origin = round(math.sqrt(n))
    if isinstance(interval, str):
        interval = round(0.1*n)
        
    print(n, origin, interval)    
    y = dataset['target']
    dataset = dataset.drop('target', axis=1)
    
    cv = 10
    optimal_k_vector = []
    lowerInterval = origin-interval
    if (origin-interval) < 1:
        lowerInterval = 1
    
    for i in range(lowerInterval, origin+interval):
        classifier.set_params(n_neighbors = i)
        scores = cross_val_score(classifier, dataset, y, cv=cv, scoring='accuracy')
        optimal_k_vector.append(scores.mean())
        
    max_value = max(optimal_k_vector)
    max_index = optimal_k_vector.index(max_value)
    
    return (origin-interval+max_index)
    
#classifier = KNeighborsClassifier(n_neighbors=8)
#print(glob.glob('*.csv'))
#df = pd.read_csv("balance_scale_clean")
#print(bruteForce(df, classifier, 'default', 'default'))
    


