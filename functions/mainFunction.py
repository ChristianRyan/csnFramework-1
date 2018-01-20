import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from report import *



def createAllModels(datasets,k,interval,searchAlgo,optimization_Type,typeOfClassifier,radius,weight):
    list_knn_models=[]
    for d in datasets:
        for i in searchAlgo:
            for j in weight:
                for l in typeOfClassifier:
                    for opt_type in optimization_Type:
                        if l=='knn': 
                            if k=='default':
                                knn = KNeighborsClassifier(n_neighbors=int(math.sqrt(d.size)), weights=j, algorithm=opt_type, leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1)
                                # contains params
                                list_knn_models.append(knn)
                            else:
                                knn = KNeighborsClassifier(n_neighbors=k, weights=j, algorithm=opt_type, leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1)
                                list_knn_models.append(knn)
                        else:
                            radiusNN=RadiusNeighborsClassifier(radius=radius, weights=j, algorithm=opt_type, leaf_size=30, p=2, metric='minkowski', outlier_label=None, metric_params=None)
                            list_knn_models.append(radiusNN)
    return list_knn_models
