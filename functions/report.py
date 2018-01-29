def mkReport(lodf):
    cols = ['Strategy', 'Dataset', 'n_instances', 'l_attributes', 'k_neighbours', 'fit_time', 'accuracy', 'f1_macro', 'f1_micro']
    finalDf = pd.DataFrame(columns=cols)
    for df in lodf:
        finalDf.append(df)

    return finalDf
