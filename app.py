import pandas as pd
import numpy as np
import sys
sys.path.append('./functions/')
from importDataset import ImportAllDatasets
from gridsearch import *
from binarySearch import *
from randomizedSearch import *
from mainFunction import *


# This is the main module the user calls

# The bulk of it goes below
def main():
    #Loading all csv files as datasets
    datasets=ImportAllDatasets()

    knn = KNeighborsClassifier(weights='distance', n_jobs=-1)
    print(knn)

    # Brute force search from 1 - sqrt(n)+bound
    # print(gridSearch([datasets[0][1]], knn))

    # Brute force search from [sqrt(n)-i, sqrt(n)+i]
    print(gridSearch([datasets[1]], knn, lwrBound = None))

    # Brute force search for custom bounds
    #print(gridSearch([datasets[0][1]], knn, lwrBound = 20, uprBound = 40))

    # Random search
    #print(randomSearch([datasets[3]],5))

    # Binary search
    #print(binarySearch(datasets[0][1],knn))

    # TODO: export report


if __name__ == "__main__":
    main()
