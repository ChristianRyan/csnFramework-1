import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
# from report import *
# from importDataset import *

# TODO: create knn classifier
# TODO: this changes based on the config file
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1)
# contains params
knnparams = knn.get_params()

# TODO: 1 optimize parameters

# TODO: 1.1 Grid search

# TODO: 1.2 Randomized search

# TODO: 1.3 OurImplementation for brute force arround sqrt(n)

# TODO: export report
