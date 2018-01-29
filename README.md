# csnFramework

Simple framework for finding the best k for kNN in python for the course in Machine Learning at Technical University Vienna.

# How to use

## Datasets

To use put clean datasets in the datasets folder with the target value named 'target'.

# k search

## Global Brute force search

Simply brute force searches from 1 - square root of n plus 10% above. Warning: Could experience long run time

## Local Brute force search

Based on the premise that the best k for kNN is in the area of sqrt(n) where n is number of instances for the dataset. The custom search allows for searching within 10% of n in the interval around sqrt(n) and brute force searching in that region.

## Randomized search

## Binary search

# Optimisation strategies



# Contributors

Nikola Jankovic

Slimane Makhlouf

Christian Ryan
