import pandas as pd
import numpy as np
import sys
sys.path.append('./functions/')
from mainFunction import knn
import ConfigReader
import importDataset


# This is the main module the user calls


# The bulk of it goes below
def main():
    #print(knn)
    #print(knn.get_params())
    
    #Loading all csv files as datasets
    datasets=ImportAllDatasets()
    
    
    # Load the configuration params
 #   k,interval,searchAlgo,optimization_Type,typeOfClassifier,radius,weight=ReadConfig()
    
    # hahahahahha u suck slim
    
# TODO: tie the functions together

# TODO: produce the report

if __name__ == "__main__":
    main()

#%%