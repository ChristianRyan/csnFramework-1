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
        n = data.shape[0]
        # Get the l
        l = data.shape[1]
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
        resultsGridSearch = pd.DataFrame(clf.cv_results_)
        # TODO: fix because cannot broadcast overwriting previous, use append
        returnDf[['fit_time', 'accuracy', 'recall_macro', 'recall_micro', 'precision_macro', 'precision_micro']] = resultsGridSearch['mean_fit_time', 'mean_test_accuracy', 'mean_test_recall_macro', 'mean_test_recall_micro', 'mean_test_precision_macro', 'mean_test_precision_micro']
        returnDf['Strategy'] = 'Brute Force'
        returnDf['Dataset'] = 'D' + str(idx)
        returnDf['n_instances'] = n
        returnDf['l_attributes'] = l
        returnDf['k_neighbours'] = 'D' + str(idx)

    return returnDf
