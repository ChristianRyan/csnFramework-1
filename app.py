import pandas as pd
import numpy as np
import sys
sys.path.append('./functions/')
from mainFunction import knn
import ConfigReader
import importDataset
import gridsearch


# This is the main module the user calls

# The bulk of it goes below
def main():

    #print(knn)
    #print(knn.get_params())

    #Loading all csv files as datasets
    datasets=ImportAllDatasets()


    # Load the configuration params
 #   k,interval,searchAlgo,optimization_Type,typeOfClassifier,radius,weight=ReadConfig()
<<<<<<< HEAD

=======
=======
    
#=======
>>>>>>> e0a6e2388b1fce9ef894cb69ddf89fe7b6a21346
    print(knn)
    print(knn.get_params())

#>>>>>>> cb3342c636f2f76fa5a32f9364282bced3792632
    # hahahahahha u suck slim


# TODO: tie the functions together

# TODO: produce the report

if __name__ == "__main__":
    main()

#%%
