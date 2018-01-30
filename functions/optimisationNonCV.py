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
    print('----KDTree----')
    global returndf
    n = dataset.shape[0]
    l = dataset.shape[1]
    k = int(math.sqrt(n))
    knnkd = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1, algorithm='kd_tree')
    y = dataset['target']
    X = dataset.drop('target', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
    start = time.time()
    knnkd.fit(X_train, y_train)
    knnkdpred = knnkd.predict(X_test)
    end = time.time()
    print(confusion_matrix(y_test, knnkdpred))

def bttreeCheck(dataset, idx):
    print('----BallTree----')
    global returndf
    n = dataset.shape[0]
    l = dataset.shape[1]
    k = int(math.sqrt(n))
    knnbt = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1, algorithm='ball_tree')
    y = dataset['target']
    X = dataset.drop('target', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
    start = time.time()
    knnbt.fit(X_train, y_train)
    knnbtpred = knnbt.predict(X_test)
    end = time.time()
    print(confusion_matrix(y_test, knnbtpred))

def bruteCheck(dataset, idx):
    print('----BruteForce----')
    global returndf
    n = dataset.shape[0]
    l = dataset.shape[1]
    k = int(math.sqrt(n))
    knnbrute = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='brute')
    y = dataset['target']
    X = dataset.drop('target', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
    start = time.time()
    knnbrute.fit(X_train, y_train)
    knnbrutepred = knnbrute.predict(X_test)
    end = time.time()
    print(confusion_matrix(y_test, knnbrutepred))

def optimizeEval2(datasets):
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
