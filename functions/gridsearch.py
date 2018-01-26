import pandas as pd
import numpy as np
import math

from sklearn.model_selection import GridSearchCV

def gridSearch(dataset, classifier, lwrBound = 1, uprBound = None):
    params = classifier.get_params()
    cols = ['Strategy', 'Dataset', 'n_instances', 'l_attributes', 'k_neighbours', 'fit_time', 'accuracy', 'recall_macro', 'recall_micro', 'precision_macro', 'precision_micro']
    returnmainDf = pd.DataFrame(columns=cols)

    returnListDfs = []
    for idx, data in enumerate(dataset):
        returnDf = pd.DataFrame(columns=cols)
        print("---Running dataset Grid Search---")

        # Get the n
        n = data.shape[0]

        # Get the l
        l = data.shape[1]

        print("Number of values for the dataset is:", n)
        print("Number of attributes for the dataset is:", l)

        # Set the upper bound for searching, default is 0.2*n
        # Can be changed to interval
        if uprBound is None and lwrBound == 1:
            uprBound = int(math.sqrt(n) + 0.2*(n))
        elif uprBound is None and lwrBound is None:
            uprBound =  int(math.sqrt(n) + 0.1*(n))
            lwrBound =  int(math.sqrt(n) - 0.1*(n))
            if lwrBound < 0:
                lwrBound = 1
        else:
            if uprBound < lwrBound:
                print('Upper bound may not be higher than lower bound')
                break

        print("Upper bound for searching is", uprBound)
        print("Lower bound for searching is", lwrBound)

        # Setting of the k for grid parameters
        gridparams = {'n_neighbors' : np.unique(np.linspace(lwrBound, uprBound, num=n).astype(int))}
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

        # Copy values from clf results
        vals = ['mean_fit_time', 'mean_test_accuracy', 'mean_test_recall_macro', 'mean_test_recall_micro', 'mean_test_precision_macro', 'mean_test_precision_micro']
        returnDf[cols[5:]] = resultsGridSearch[vals]
        # Fill with rest of data
        returnDf['Strategy'] = 'Brute Force'
        returnDf['Dataset'] = 'D' + str(idx)
        returnDf['n_instances'] = n
        returnDf['l_attributes'] = l
        returnDf['k_neighbours'] = gridparams['n_neighbors']
        # Append to a list and iterate later over list to append to single data frame
        returnListDfs.append(returnDf)

    # Looping over list to create single dataframe
    for dataf in returnListDfs:
        returnmainDf = returnmainDf.append(dataf)

    return returnmainDf
