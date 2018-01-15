import pandas as pd
import glob

def ImportAllDatasets():
    List_DataSets=[]
    List_files=glob.glob("Datasets/*.csv")
    for i in List_files:
        df = pd.read_csv(i)
        print(df)
        List_DataSets.append(df)
    return List_DataSets


List=ImportAllDatasets()
#%%
    