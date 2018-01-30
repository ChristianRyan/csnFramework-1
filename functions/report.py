import pandas as pd

def mkReport(lodf):
    cols = ['Strategy', 'Dataset', 'n_instances', 'l_attributes', 'k_neighbours', 'fit_time', 'accuracy', 'f1_macro', 'f1_micro']
    finalDf = pd.DataFrame(columns=cols)
    for df in lodf:
        finalDf.append(df, ignore_index=True)

    finalDf.reset_index(drop=True, inplace=True)
    return finalDf
