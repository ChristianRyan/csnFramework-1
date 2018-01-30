import pandas as pd
import numpy as np
import sys
sys.path.append('./functions/')
from importDataset import ImportAllDatasets
from gridsearch import *
from binarySearch import *
from randomizedSearch import *
from report import *
from sklearn.neighbors import KNeighborsClassifier
from optimisation import *


# This is the main module the user calls

# The bulk of it goes below
def main():
    #Loading all csv files as datasets
    datasets=ImportAllDatasets()

    knn = KNeighborsClassifier(weights='distance', n_jobs=-1)
    print(knn)

    # Brute force search from 1 - sqrt(n)+bound
    # gs2 = gridSearch([datasets[1]], knn)

    # Brute force search from [sqrt(n)-i, sqrt(n)+i]
    # gs1 = gridSearch(datasets, knn, uprBound=None, lwrBound = None)

    # Brute force search for custom bounds
    gs3 = gridSearch(datasets, knn, lwrBound = 10, uprBound = 10)

    # Random search

    rs1 = randomSearch(datasets, 25)

    # Binary search
    bslist = []
    for dataset in datasets:
        try:
            bslist.append(binarySearch(dataset, knn))
        except Exception as e:
            continue

    bs1 = mkReport(bslist)

    # Create single dataframe
    lodf = [gs3, rs1, bs1]
    finalDf = mkReport(lodf)
    finalDf.to_csv('./results.csv')

    # Optimisation strategies
    oe = optimizeEval(datasets)
    oe.to_csv('./optimisation.csv')

if __name__ == "__main__":
    main()
