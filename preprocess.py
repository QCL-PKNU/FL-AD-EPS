import os
import sys
import pandas as pd
import numpy as np
from src.datafolders import *


def normalize(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a


def load_data(eps):
	dataset = 'EPS'
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)

	dataset_folder = 'data/EPS'
	file = os.path.join(dataset_folder, f'{eps}.csv')
	df = pd.read_csv(file, header=0)

	lbl = df['ANOMALY']
	df = df.drop(['ANOMALY'], axis=1)

	split = int(len(df)*0.7) # 70% train, 30% test

	df_train = df.iloc[:split]
	df_test = df.iloc[split:]
	lbl = lbl.iloc[split:]

	print(lbl.value_counts())
	# 0 - Normal, 1 - Anomaly

	train, min_a, max_a = normalize(df_train.values)
	test, _, _ = normalize(df_test.values, min_a, max_a)

	labels = np.zeros_like(test)
   
	# Convert one column to 3 columns of anomaly labels to match the test shape
	for i in range(len(lbl)):
		if lbl.iloc[i] == 1:
			labels[i, :] = 1
		else:
			labels[i, :] = 0

	# First COLUMN is the Speed, Second COLUMN is the Angle, Third COLUMN is the Torque
	print("Train Shape: " + str(train.shape), "Test Shape: " + str(test.shape), "Labels Shape: " + str(labels.shape))

	for file in ['train', 'test', 'labels']:
		np.save(os.path.join(folder, f'{file}.npy'), eval(file)) # Save as NPY file for faster loading


if __name__ == '__main__':
	# Command line arguments
	args = sys.argv

	if len(args) < 2:
		print("Please specify the dataset to be processed. Options: eps, eps1, eps2, eps3, eps4, or eps5")
		exit(0)

	if args[1] == 'eps1':
		load_data("eps1")
	elif args[1] == 'eps2':
		load_data("eps2")
	elif args[1] == 'eps3':
		load_data("eps3")
	elif args[1] == 'eps4':
		load_data("eps4")
	elif args[1] == 'eps5':
		load_data("eps5")
	else:
		load_data("eps")