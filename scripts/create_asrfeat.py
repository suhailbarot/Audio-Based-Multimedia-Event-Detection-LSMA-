#!/bin/python
import numpy as np
from scipy.sparse import csr_matrix,save_npz
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from nltk import word_tokenize as tokenize
from nltk.corpus import stopwords

import glob
from collections import defaultdict,Counter
from tqdm import tqdm

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print "Usage: {0} vocab_file, file_list".format(sys.argv[0])
		print "vocab_file -- path to the vocabulary file"
		print "file_list -- the list of videos"
		exit(1)
	tfidf = TfidfVectorizer()
	stopword_set = stopwords.words('english')
	vocabulary = set()
	asr_file_list = glob.glob('../11775_asr/*.txt')
	asr_file_set = set(glob.glob('../11775_asr/*.txt'))
	corpus = []
	word_corpus = []
	for trans_file in tqdm(asr_file_list):
		f = open(trans_file)
		corpus.append(f.read())
		word_corpus.append(tokenize(f.read()))
	X = tfidf.fit_transform(corpus)
	print X.shape
	print len(word_corpus),len(word_corpus[1])
	dic = tfidf.vocabulary_
	file_list = open(sys.argv[2])



	# i=0
	# j=0
	# for line in tqdm(file_list.readlines()):
	# 	if "../11775_asr/"+line.strip()+".txt" not in asr_file_set:
	# 		outfile = open('asrfeat/'+line.strip()+'.txt','w+')
	# 		feature_bog = [0]*X.shape[1]
	#  		outfile.write(" ".join([str(x) for x in feature_bog]))
	#  		j+=1


	# 	else:
	# 		feature_bog = list(X[i,:].toarray()[0])
			
	# 		outfile = open('asrfeat/'+line.strip()+'.txt','w+')
	#  		outfile.write(" ".join([str(x) for x in feature_bog]))
	#  		i+=1
	# print i+j







	# for i,trans_file in tqdm(enumerate(asr_file_list)):
	# 	for word in tokenize(open(trans_file).read().lower()):
	# 		if word in dic:
	# 			if word not in stopword_set and len(word)>2 and X[i,dic[word]]>0.15:
				# 	vocabulary.add(word)
	# 		# vocabulary.update(tokenize(line.strip()))
	# word_to_index = {}
	# for i,word in enumerate(vocabulary):
	# 	word_to_index[word] = i
	# print len(vocabulary)
	c = Counter()
	for i,trans_file in tqdm(enumerate(asr_file_list)):
		for word in tokenize(open(trans_file).read().lower()):
			if word not in stopword_set:
				c[word]+=1
	n = 7500
	print c
	vocabulary.update([x for x,y in c.most_common()[0:2000]])
	word_to_index = {}
	for i,word in enumerate(vocabulary):
		word_to_index[word] = i
	print len(vocabulary)
	vocab_size = len(vocabulary)

	file_list = open(sys.argv[2])
	data = []
	for line in tqdm(file_list.readlines()):
		if "../11775_asr/"+line.strip()+".txt" not in asr_file_list:
			# data.append([0]*vocab_size)
			feature_bog = np.array([1/float(vocab_size)]*vocab_size)
			outfile = open('asrfeat/'+line.strip()+'.txt','w+')
			outfile.write(" ".join([str(x) for x in feature_bog]))
		else:
			feature_bog = [0]*vocab_size
			for line1 in open("../11775_asr/"+line.strip()+".txt").readlines():
				for word in tokenize(line1.strip()):
					if word in vocabulary:
						feature_bog[word_to_index[word]]+=1
			# feature_bog = [1/float(x) if x!=0 else 0 for x in feature_bog]
			outfile = open('asrfeat/'+line.strip()+'.txt','w+')
			s = sum(feature_bog)
			if s==0:
				outfile.write(" ".join([str(1/float(vocab_size)) for x in feature_bog]))
			else:
				outfile.write(" ".join([str(x/float(s)) for x in feature_bog]))

	# 		# data.append(temp[:])

	# data_array_sparse = csr_matrix(np.array(data))
	# out_file = open('asr_data.npz','w+')
	# save_npz(out_file,data_array_sparse)








	print "ASR features generated successfully!"
