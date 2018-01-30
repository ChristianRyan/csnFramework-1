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
    #gs2 = gridSearch([datasets[1]], knn)

    # Brute force search from [sqrt(n)-i, sqrt(n)+i]
    gs1 = gridSearch(datasets, knn, uprBound=None, lwrBound = None)

    # Brute force search for custom bounds
    #print(gridSearch([datasets[0][1]], knn, lwrBound = 20, uprBound = 40))

    # Random search
<<<<<<< HEAD
#    rs1 = randomSearch([datasets[1]],2)
#    print(rs1)
=======
    rs1 = randomSearch(datasets, 20)
>>>>>>> 1fe5c8c1a3119f86294cd500dbe49083bf68da7c

    # Binary search
    bs1 = binarySearch(datasets, knn)

    # Create single dataframe
    lodf = [gs1, rs1, bs1]
    finalDf = mkReport(lodf)
    finalDf.to_csv('./results.csv')

    # Optimisation strategies
    oe = optimizeEval(datasets)
    oe.to_csv('./optimisation.csv')

if __name__ == "__main__":
    main()
