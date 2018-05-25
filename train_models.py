#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import read_yale
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import time
import os
import bz2

code_folder = os.getcwd()
model_folder = code_folder + r'\Trainded_Models'

k_fold = 10 # CHANGE TO 10 IN FINAL VERSION!!!


def main():
	print("Initializing...")
	start = time.time()
	images, resolution = read_yale.get_croppedyale_as_df()
	#We leave 20 % of the data for final validation. This is not used in training
	final_validation = images.sample(frac=0.2)
	images = images.loc[~images.index.isin(final_validation.index)]
	final_validation_labels = final_validation.index.get_level_values('person').values
	final_pic_names = final_validation.index.get_level_values('pic_name').values
	final_validation = final_validation.values

	total_training = images
	total_labels = total_training.index.get_level_values('person').values
	total_training = total_training.values


	print("Loaded dataset and separated final validation data")
	print("Time passed: " + str(time.time()-start))
	#We use 10-fold cross-validation. The accuracy of the 10 SVM's is then calculated on majority vote on final 
	#validation data
	k_fold_data = {}
	k_fold_labels = {}
	print("Separating remaining data into 10 subsets...")
	#SPLIT TRAINING DATA TO 10 SUBSETS RANDOMLY
	for i in range(k_fold):
		k_fold_data[i] = images.sample(frac=1/k_fold)
		images = images.loc[~images.index.isin(k_fold_data[i].index)]
		k_fold_labels[i] = k_fold_data[i].index.get_level_values('person').values
		k_fold_data[i] = k_fold_data[i].values

	print("Data ready.")
	print("Time passed: " + str(time.time()-start))
	#TRAIN 10 SVMs
	print("Training SVMs...")
	accuracies = []
	SVMs = {}
	for i in range(k_fold):
		print("Training model: " + str(i+1)+ "...")
		formatted = False
		for key in k_fold_data:
			if key == i:
				validation_data = k_fold_data[i]
				validation_labels = k_fold_labels[i]
			elif not formatted:
				training_data = k_fold_data[i]
				training_labels = k_fold_labels[i]
				formatted = True
			else:
				training_data = np.vstack([training_data, k_fold_data[i]])
				training_labels = np.append(training_labels, k_fold_labels[i])
		clf = svm.LinearSVC()
		#print(training_data)
		#print(training_labels)
		clf.fit(training_data,training_labels)
		acc_pred = clf.predict(validation_data)
		acc = sklearn.metrics.accuracy_score(validation_labels,acc_pred)
		accuracies.append(acc)
		SVMs[i] = clf
		with open(model_folder + r'\svm_'+str(i)+'.pickle','wb') as file:
			pickle.dump(clf,file)
		print("Model "+ str(i+1)+" trained!")
		print("Time passed: " + str(time.time()-start))

	with open(model_folder+r'\accuracies.pickle','wb') as f:
		pickle.dump(accuracies,f)
	print("Cross validation ready. Training final model...")
	clf = svm.LinearSVC()
	clf.fit(total_training,total_labels)
	with open(model_folder + r'\svm_total.pickle','wb') as file:
		pickle.dump(clf,file)

	print("All models trained!")

	with bz2.BZ2File(model_folder + r'\test_set.pbz2','w') as file:
		pickle.dump(final_validation,file)
	with bz2.BZ2File(model_folder + r'\test_labels.pbz2','wb') as file:
		pickle.dump(final_validation_labels,file)
	with bz2.BZ2File(model_folder + r'\test_pic_names.pbz2','wb') as file:
		pickle.dump(final_pic_names,file)

if __name__ == "__main__":
	main()


