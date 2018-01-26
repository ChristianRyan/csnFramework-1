import pandas as pd
import glob


def ImportAllDatasets():
    """Importing all the csv files from the dataset directory"""
    List_DataSets=[]
    List_files=glob.glob("./Datasets/*.csv")
    print("Importing following data: ", List_files)
    for i in List_files:
        print("Importing " +i)
        df = pd.read_csv(i)
        List_DataSets.append(df)

    return List_DataSets



#%%
