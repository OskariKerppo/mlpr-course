#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 22:39:40 2018

@author: mika

This is a simple example where we read the images as a DataFrame using
get_croppedyale_as_df() and then select some rows of the DataFrame based
on person 'names' or lighting conditions.
"""

import read_yale
from sklearn import svm
import numpy as np
import pandas as pd

images, resolution = read_yale.get_croppedyale_as_df()
# The data frame uses a MultiIndex to store the person 'names' and lighting
# conditions. Here we briefly demonstrate using the data frame.
#print(images.columns.values)
#print(len(images))
# Get the names of the persons
row_persons = images.index.get_level_values('person')
#print(row_persons)
#print(row_persons)
# Get all images of 'yaleB10'
rows_include = (row_persons == 'yaleB10')
pics_B10 = images[rows_include]
rows_include = (row_persons == 'yaleB11')
pics_B11 = images[rows_include]
#print(pics_B10)
#print(pics_B10) # there are over 30 000 columns so results are not pretty..
# Get all images under conditions "P00A-130E+20"
#row_conds = images.index.get_level_values('pic_name')
#rows_include = (row_conds == 'P00A-130E+20')
#pics_2 = images[rows_include]
#print(pics_2)

sample = pics_B10.append(pics_B11)
#print(sample)
X = sample.sample(frac=0.7)
validation_data = sample.loc[~sample.index.isin(X.index)]
validation_labels = validation_data.index.get_level_values('person').values
validation_data = validation_data.values
#print(X)
y = X.index.get_level_values('person').values
#print(y)
X = X.values


clf = svm.SVC()
clf.fit(X,y)

predictions = clf.predict(validation_data)
print(predictions)
print(validation_labels)

correct = 0
incorrect = 0
for i in range(len(predictions)):
	if predictions[i] == validation_labels[i]:
		correct += 1
	else:
		incorrect += 1

print("Correctly classified: " + str(correct))
print("Percentage correct: " + str(correct/(correct + incorrect)))



