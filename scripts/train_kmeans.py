#!/bin/python 

import numpy as np
import os
from sklearn.cluster.k_means_ import MiniBatchKMeans,KMeans
import cPickle
import sys
import csv
from tqdm import tqdm

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
	print "e"
	if len(sys.argv) != 4:
		print "Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0])
		print "mfcc_csv_file -- path to the mfcc csv file"
		print "cluster_num -- number of cluster"
		print "output_file -- path to save the k-means model"
		exit(1)
	mfcc_csv_file = open(sys.argv[1]) 
	output_file = sys.argv[3]
	cluster_num = int(sys.argv[2])
	kmeans = MiniBatchKMeans(n_clusters=cluster_num)
	reader = csv.reader(mfcc_csv_file)
	inp_list = [] 
	for line in tqdm(reader):
		inp_list.append(map(float,line[0].split(';')))
		# print map(float,line[0].split(';'))
	# inp_array = np.array(inp_list[:100]).reshape(-1,1)
	data_arr = np.array(inp_list)
	print data_arr
	kmeans.fit(data_arr)
	with open(output_file+'.pkl', 'wb') as fid:
		cPickle.dump(kmeans, fid)


	print "K-means trained successfully!"
