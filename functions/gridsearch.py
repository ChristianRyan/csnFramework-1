import pandas as np
import numpy as np
import math

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from importDataset import *

def grdSearch(dataset, classifier, gridparams = None, lwrBound = 1):
    params = classifier.get_params()

    returnDf = pd.DataFrame(columns = ['Strategy', 'Dataset', 'n_instances', 'l_attributes', 'k_neighbours', 'fit_time', 'accuracy', 'recall_macro', 'recall_micro', 'precision_macro', 'precision_micro'])
    for idx, data in enumerate(dataset):
        print("---Running dataset Grid Search---")
        # Get the n
        n = int(data.describe().iloc[0,0])
        print("n for the dataset is", n)
        # Set the upper bound for searching, default is 0.2*n
        uprBound = int(math.sqrt(n) + 0.2*(math.sqrt(n)))
        print("Upper bound for searching is", uprBound)
        # Setting of the k for grid parameters
        gridparams = {'n_neighbors' : np.unique(np.linspace(1, uprBound, num=n).astype(int))}
        print("Grid parameters for the dataset are", gridparams)
        # setting the scores and GridSearchCV
        scores = ['accuracy', 'recall_macro', 'recall_micro', 'precision_macro', 'precision_micro']
        clf = GridSearchCV(classifier, gridparams, n_jobs=-1, scoring=scores, refit=False) # BE careful with cores :D
        print(clf)
        # Seperating the X and y, the class to be classified
        y = data['target']
        X = data.drop('target', axis=1)
        print("---Fitting data Grid Search---")
        clf.fit(X, y)
        # Results of the fit
        resultsdf = pd.DataFrame(clf.cv_results_)
        returnDict['dataset '+idx] = ''
        #lodf.append(pd.DataFrame(clf.cv_results_))
    # For now returns a list of dataframes that contain grid search result parameters (or rather should didnt test), should return optimal k
    return returnDict
