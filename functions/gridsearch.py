import pandas as np
import numpy as np
import math

from sklearn.model_selection import GridSearch

def grdSearch(dataset, classifier, gridparams = None, upperadd = 50, number_of_values = 100):
    params = classifier.get_params()

    lodf = []
    for data in dataset:
        n = int(data.describe().iloc[0,0])
        paramgrid = {'n_neighbors' : np.linspace(1, round(math.sqrt(n)) + upperadd, 5).astype(int)}
        clf = GridSearchCV(classifier, paramgrid, n_jobs=-1) # TODO: BE careful with cores :D
        y = data['target']
        X = dataset.drop('target', axis=1)
        clf.fit(X, y)
        lodf.append(pd.DataFrame(clf.cv_results_))
    # TODO: For now returns a list of dataframes that contain grid search result parameters (or rather should didnt test), should return optimal k
    return lodf
