import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from report import *

# TODO: create knn classifier
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1)

# TODO: optimize parameters

# TODO: export report
