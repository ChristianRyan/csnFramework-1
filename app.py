import pandas as pd
import numpy as np
import sys
sys.path.append('./functions/')
from mainFunction import knn
import ConfigReader
from importDataset import *
import gridsearch


# This is the main module the user calls

# The bulk of it goes below
def main():
    #Loading all csv files as datasets
    datasets=ImportAllDatasets()


    # Load the configuration params
    k,interval,searchAlgo,optimization_Type,typeOfClassifier,radius,weight=ReadConfig()

    print(knn)
    print(knn.get_params())

# TODO: tie the functions together

# TODO: produce the report

if __name__ == "__main__":
    main()

#%%
