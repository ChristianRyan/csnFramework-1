import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time
import math
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import *

cols = ['Strategy', 'Dataset', 'n', 'l', 'k', 'avg_fit_time', 'avg_score_time', 'cv_runtime']
global returndf
returndf =  pd.DataFrame(columns=cols)

def kdtreeCheck(dataset, idx):
    global returndf
    n = dataset.shape[0]
    l = dataset.shape[1]
    k = int(math.sqrt(n))
    knnkd = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1, algorithm='kd_tree')
    y = dataset['target']
    X = dataset.drop('target', axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
    start = time.time()
    # knnkd.fit(X_train, y_train)
    # knnkdpred = knnkd.predict(X_test)
    scoring=['accuracy', 'f1_macro', 'f1_micro']
    resultskd = cross_validate(knnkd, X, y, scoring=scoring)
    resultskddf = pd.DataFrame(resultskd)
    end = time.time()
    returndf = pd.concat([returndf, pd.DataFrame.from_records([{
        'Strategy': 'KDTree',
        'Dataset': idx,
        'n': n,
        'l': l,
        'k': k,
        'avg_fit_time': resultskddf['fit_time'].mean(),
        'avg_score_time': resultskddf['score_time'].mean(),
        'cv_runtime': end - start,
        'accuracy':resultskddf['test_accuracy'].mean(),
        'f1_macro':resultskddf['test_f1_macro'].mean(),
        'f1_micro':resultskddf['test_f1_micro'].mean()
    }])])

def bttreeCheck(dataset, idx):
    global returndf
    n = dataset.shape[0]
    l = dataset.shape[1]
    k = int(math.sqrt(n))
    knnbt = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1, algorithm='ball_tree')
    y = dataset['target']
    X = dataset.drop('target', axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
    start = time.time()
    # knnbt.fit(X_train, y_train)
    # knnbtpred = knnbt.predict(X_test)
    scoring=['accuracy', 'f1_macro', 'f1_micro']
    resultsbt = cross_validate(knnbt, X, y, scoring=scoring)
    resultsbtdf = pd.DataFrame(resultsbt)
    end = time.time()
    returndf = pd.concat([returndf, pd.DataFrame.from_records([{
        'Strategy': 'BallTree',
        'Dataset': idx,
        'n': n,
        'l': l,
        'k': k,
        'avg_fit_time': resultsbtdf['fit_time'].mean(),
        'avg_score_time': resultsbtdf['score_time'].mean(),
        'cv_runtime': end - start,
        'accuracy':resultsbtdf['test_accuracy'].mean(),
        'f1_macro':resultsbtdf['test_f1_macro'].mean(),
        'f1_micro':resultsbtdf['test_f1_micro'].mean()
    }])])

def bruteCheck(dataset, idx):
    global returndf
    n = dataset.shape[0]
    l = dataset.shape[1]
    k = int(math.sqrt(n))
    knnbrute = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='brute')
    y = dataset['target']
    X = dataset.drop('target', axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
    start = time.time()
    #knnbrute.fit(X_train, y_train)
    #knnbrutepred = knnbrute.predict(X_test)
    scoring=['accuracy', 'f1_macro', 'f1_micro']
    resultsbrute = cross_validate(knnbrute, X, y, scoring=scoring)
    resultsbrutedf = pd.DataFrame(resultsbrute)
    end = time.time()
    returndf = pd.concat([returndf, pd.DataFrame.from_records([{
        'Strategy': 'Brute Force',
        'Dataset': idx,
        'n': n,
        'l': l,
        'k': k,
        'avg_fit_time': resultsbrutedf['fit_time'].mean(),
        'avg_score_time': resultsbrutedf['score_time'].mean(),
        'cv_runtime': end - start,
        'accuracy':resultsbrutedf['test_accuracy'].mean(),
        'f1_macro':resultsbrutedf['test_f1_macro'].mean(),
        'f1_micro':resultsbrutedf['test_f1_micro'].mean()
    }])])


def optimizeEval(datasets):
    for idx, dataset in enumerate(datasets):
        print("Dataset", idx)
        n = dataset.shape[0]
        print("n for dataset is", n)
        l = dataset.shape[1]
        print("l for dataset is", l)
        try:
            kdtreeCheck(dataset, idx)
            bttreeCheck(dataset, idx)
            bruteCheck(dataset, idx)
        except ValueError as e:
            continue
        except MemoryError as f:
            continue

    return returndf
