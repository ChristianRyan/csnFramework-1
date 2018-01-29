import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

def kdtreeCheck(dataset):
    n = dataset.shape[0]
    k = int(math.sqrt(n))
    knnkd = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1, algorithm='kd_tree')
    y = dataset['target']
    X = dataset.drop('target', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
    start = time.time()
    knnkd.fit(X_train, y_train)
    knnkdpred = knnkd.predict(X_test)
    end = time.time()
    print('Accuracy: ', accuracy_score(y_test, knnkdpred))
    print('F1 score: ', f1_score(y_test, knnkdpred, average='weighted'))
    print('Runtime was ', end - start)

def bttreeCheck(dataset):
    n = dataset.shape[0]
    k = int(math.sqrt(n))
    knnbt = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1, algorithm='ball_tree')
    y = dataset['target']
    X = dataset.drop('target', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
    start = time.time()
    knnbt.fit(X_train, y_train)
    knnbtpred = knnbt.predict(X_test)
    end = time.time()
    print('Accuracy: ', accuracy_score(y_test, knnbtpred))
    print('F1 score: ', f1_score(y_test, knnbtpred, average='weighted'))
    print('Runtime was ', end - start)


def bruteCheck(dataset):
    n = dataset.shape[0]
    k = int(math.sqrt(n))
    knnbrute = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='brute')
    y = dataset['target']
    X = dataset.drop('target', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
    start = time.time()
    knnbrute.fit(X_train, y_train)
    knnbrutepred = knnbrute.predict(X_test)
    end = time.time()
    print('Accuracy: ', accuracy_score(y_test, knnbrutepred))
    print('F1 score: ', f1_score(y_test, knnbrutepred, average='weighted'))
    print('Runtime was ', end - start)

def optimizeEval(datasets):
    for idx, dataset in enumerate(datasets):
        print("Dataset", idx)
        n = dataset.shape[0]
        print("n for dataset is", n)
        l = dataset.shape[1]
        print("l for dataset is", l)
        try:
            kdtreeCheck(dataset)
            bttreeCheck(dataset)
            bruteCheck(dataset)
        except ValueError as e:
            print(e)
            continue
        except MemoryError as f:
            print('MemoryError')
            continue
