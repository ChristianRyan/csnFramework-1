import pandas as np
import numpy as np
import math

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from importDataset import *

def grdSearch(dataset, classifier, gridparams = None, upperadd = 50, number_of_values = 25):
    params = classifier.get_params()

    returnDict = {}
    for idx, data in enumerate(dataset):
        print("---Running dataset Grid Search---")
        n = int(data.describe().iloc[0,0])
        print("n for the dataset is", n)
        gridparams = {'n_neighbors' : np.linspace(1, round(math.sqrt(n)) + upperadd, number_of_values).astype(int)}
        print("Grid parameters for the dataset are", gridparams)
        clf = GridSearchCV(classifier, gridparams, n_jobs=-1) # TODO: BE careful with cores :D
        print(clf)
        y = data['target']
        X = data.drop('target', axis=1)
        print("---Fitting data Grid Search---")
        clf.fit(X, y)
        returnDict['dataset '+idx] = ''
        #lodf.append(pd.DataFrame(clf.cv_results_))
    # For now returns a list of dataframes that contain grid search result parameters (or rather should didnt test), should return optimal k
    return returnDict
