import pandas as pd
import numpy as np
import sys
sys.path.append('./functions/')
#from mainFunction import knn
from ConfigReader import ReadConfig
from importDataset import ImportAllDatasets
from gridsearch import *
<<<<<<< HEAD
from binarySearch import *
#from randomizedSearch import *
=======
from randomizedSearch import *
>>>>>>> a1b195f12b231fd7443d4fe658719c9936f2fb24
from mainFunction import *


# This is the main module the user calls

# The bulk of it goes below
def main():
    #Loading all csv files as datasets
    datasets=ImportAllDatasets()
#    print(datasets[3])

    # Load the configuration params
    k,interval,searchAlgo,optimization_Type,typeOfClassifier,radius,weight=ReadConfig()

    # TODO: create knn classifier
    # For now I have a placeholder
<<<<<<< HEAD
    knn = KNeighborsClassifier(n_neighbors=100, n_jobs=-1)
    print(knn)
    
=======
    #knn = KNeighborsClassifier(n_neighbors=100, n_jobs=-1)
    #print(knn)
>>>>>>> a1b195f12b231fd7443d4fe658719c9936f2fb24
    # TODO: this changes based on the config file

    # Brute force search from 1 - sqrt(n)+bound
    # print(gridSearch([datasets[0][1]], knn))

    # Brute force search from [sqrt(n)-i, sqrt(n)+i]
    gs = gridSearch(datasets[0], knn, lwrBound = None)
    gs.to_csv('gridsearchlarge.csv')


    # TODO: 1.1 Grid search
    #print(gridSearch([datasets[0][1]], knn))
    # for value in datasets:
    #     print(type(value))
    #     for thing in value:
    #         print(type(thing))

    # Brute force search for custom bounds
    #print(gridSearch([datasets[0][1]], knn, lwrBound = 20, uprBound = 40))


    # TODO: Randomized search

<<<<<<< HEAD
    # TODO: Binary search
    print(binarySearch(datasets[0][1],knn))
=======


>>>>>>> a1b195f12b231fd7443d4fe658719c9936f2fb24
    # TODO: export report


   # print(knn)
    #print(knn.get_params())

    #lodf = grdSearch(datasets, knn)
    #print(lodf)
    
    print(randomSearch([datasets[3]],5))


# TODO: tie the functions together

# TODO: produce the report

if __name__ == "__main__":
    main()

#%%
