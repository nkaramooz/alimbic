import pandas as pd
import re
import psycopg2
from sqlalchemy import create_engine
from nltk.stem.wordnet import WordNetLemmatizer
from operator import itemgetter
import nltk.data
import numpy as np
import time
import utilities.utils as u, utilities.pglib as pg
import collections
import math
import os
import random
from tempfile import gettempdir
import zipfile
import sys
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import snomed_annotator as ann
import pickle as pk
from keras.datasets import imdb
import sys
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import load_model


vocabulary_size = 50000

# need one for root (condition), need one for rel treatment
# need one for none (spacer)
vocabulary_spacer = 3
# [root, rel, spacer]

def get_word_counts():
	query = "select sentence_tuples from annotation.sentences"
	conn,cursor = pg.return_postgres_cursor()
	sentence_df = pg.return_df_from_query(cursor, query, None, ['sentence_tuples'])
	sentences = sentence_df['sentence_tuples'].tolist()
	counts = {}

	for i,s in sentence_df.iterrows():
		for words in s['sentence_tuples']:
			words = words.strip('(')
			words = words.strip(')')
			words = tuple(words.split(","))
			
			if words[0] in counts.keys():
				counts[words[0]] = counts[words[0]] + 1
			else:
				counts[words[0]] = 1
	counts = OrderedDict(sorted(counts.items(), key=itemgetter(1), reverse = True))

	with open('word_count.pickle', 'wb') as handle:
		pk.dump(counts, handle, protocol=pk.HIGHEST_PROTOCOL)

	return counts

def read_data():
	query = "select sentence_tuples from annotation.sentences"
	conn,cursor = pg.return_postgres_cursor()
	sentence_df = pg.return_df_from_query(cursor, query, None, ['sentence_tuples'])
	sentences = sentence_df['sentence_tuples'].tolist()
	word_list = []

	for i,s in sentence_df.iterrows():
		for words in s['sentence_tuples']:
			words = words.strip('(')
			words = words.strip(')')
			words = tuple(words.split(","))
			
			word_list.append(words[0])
	
	return word_list

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

  with open('reversed_dictionary.pickle', 'wb') as rd:
  	pk.dump(reversed_dictionary, rd, protocol=pk.HIGHEST_PROTOCOL)

  with open('dictionary.pickle', 'wb') as di:
  	pk.dump(dictionary, di, protocol=pk.HIGHEST_PROTOCOL)

  return data, count, dictionary, reversed_dictionary


def build():

	vocabulary = read_data()

	data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
	                                                            vocabulary_size-vocabulary_spacer)
	del vocabulary


def load_word_counts_dict():
	with open('word_count.pickle', 'rb') as handle:
		counts = pickle.load(handle)
		print(counts[0:(vocabulary_size-vocabular_spacer)])

# How does training 
#[[X (i.e. condition), Y (i.e. treatment), label]]
# Will need to get all children of X, and all children of Y
labeled_set = [['22298006','387458008', '1'], #aspirin
	['22298006', '33252009', '1'], # beta blocker
	['22298006', '734579009', '1'], # ace inhibitor
	['44054006', '7947003', '0'],
	['44054006', '400556009', '1'],
	['44054006', '418285008', '0'],
	['44054006', '48698004', '0'],
	['44054006', '7092007', '0'],
	['44054006', '386879008', '1'],
	['44054006', '63718003', '0'],
	['44054006', '391858005', '0'],
	['44054006', '387419003', '1'],
	['44054006', '387124009', '0'],
	['44054006', '308111008', '0'],
	['44054006', '1182007', '0'],
	['44054006', '308113006', '0'],
	['44054006', '96302009', '0'],
	['44054006', '102747008', '0'],
	['44054006', '308111008', '0'],
	['44054006', '44054006', '0'],
	['44054006', '38341003', '0'],
	['44054006', '230690007','0'],
	['34000006', '68887009', '1'],
	['34000006', '395726003', '1'],
	['34000006', '386835005', '1'],
	['34000006', '414805007', '1'],
	['34000006', '372574004', '1'],
	['34000006', '387248006', '1'],
	['34000006', '79440004', '1'],
	['34000006', '363779003', '0'],
	['34000006', '271737000', '0'],
	['34000006', '14189004', '0'],
	['34000006', '24526004', '0'],
	['25374005', '346712003', '0'],
	['25374005', '398866008', '1'],
	['25374005', '108418007 ', '1'],
	['93880001', '119746007', '1'],
	['93880001', '367336001', '1'],
	['93880001', '261479009', '0'],
	['363418001', '367336001', '1'],
	['363418001', '386920008', '1'],
	['363418001', '450556004', '0'],
	['13645005', '79440004', '1'],
	['13645005', '428311008', '1'],
	['13645005', '59187003', '0']] 

def get_data():
	conn,cursor = pg.return_postgres_cursor()
	labelled_set = pd.DataFrame()
	results = pd.DataFrame()
	for index,item in enumerate(labeled_set):
		print(item[2])
		root_cids = [item[0]]
		root_cids.extend(ann.get_children(item[0], cursor))

		rel_cids = [item[1]]
		rel_cids.extend(ann.get_children(item[1], cursor))


		# Want to select sentenceIds that contain both
		sentence_query = "select sentence_tuples from annotation.sentences where conceptid in %s and id in (select id from annotation.sentences where conceptid in %s) limit 1000"

		#this should give a full list for the training row
		sentences = pg.return_df_from_query(cursor, sentence_query, (tuple(root_cids), tuple(rel_cids)), ["sentence_tuples"])
		# Now you get the sentence, but you need to remove root and replace with word index
		# Create sentence list
		# each sentence list contains a list vector for word index
		label = item[2]
	
		for index,sentence in sentences.iterrows():

			sentence_root_index = None
			sentnece_rel_index = None
			for index,words in enumerate(sentence['sentence_tuples']):

				words = words.strip('(')
				words = words.strip(')')
				words = tuple(words.split(","))	
				
				# Will only handle one occurrence of each conceptid for simplification
				if words[1] in root_cids:
					sentence_root_index = index
				elif words[1] in rel_cids:
					sentence_rel_index = index


				# while going through, see if conceptid associated then add to dataframe with root index and rel_type index
			results = results.append(pd.DataFrame([[sentence, sentence_root_index, sentence_rel_index, label]], columns=['sentence_tuples', 'root_index', 'rel_index', 'label']))
	results.to_pickle("./sample_train.pkl")

def serialize_data():
	sample_train = pd.read_pickle("./sample_train.pkl")
	with open('reversed_dictionary.pickle', 'rb') as rd:
  		reverse_dictionary = pk.load(rd)

	  	with open('dictionary.pickle', 'rb') as d:
	  		dictionary = pk.load(d)
	

	max_ln_arr = []
	for index, sentence in sample_train.iterrows():
		for index, tup in sentence['sentence_tuples'].iteritems():
			max_ln_arr.append(len(tup))
	
	## this is how much should be padded
	max_words = max(max_ln_arr)
	print("max_ln: " + str(max_words))
	x_train = []
	y_train = []

	for index, sentence in sample_train.iterrows():
		root_index = sentence['root_index']
		rel_index = sentence['rel_index']
		root_cid = None
		rel_cid = None
		# unknowns will have integer of 50000
		sample = [(vocabulary_size-1)]*max_words

		label = sentence['label']
		for index, tup in sentence['sentence_tuples'].iteritems():
			for ind, t in enumerate(tup):
				t = t.lower()
	
				t = t.strip('(')
				t = t.strip(')')
				t = tuple(t.split(","))	


				if ind == root_index and root_cid == None:
					root_cid = t[1]
					sample[ind] = vocabulary_size-3
					# append index for root
				elif ind == rel_index and rel_cid == None:
					rel_cid = t[1]
					sample[ind] = vocabulary_size-2
					# append index for rel
				elif t[1] == root_cid:
					sample[ind] = vocabulary_size-2
				elif t[1] == rel_index:
					sample[ind] = vocabulary_size-2
				elif t[0] in dictionary.keys():
					sample[ind] = dictionary[t[0]]
				else:
					sample[ind] = dictionary['UNK']
		x_train.append(sample)
		y_train.append(label)

	print(x_train)
	print(y_train)
	print(dictionary['UNK'])
	return x_train, y_train, max_words


def build_model():
	x_train, y_train, max_words = serialize_data()
	# x_train = sequence(x_train)
	x_train = np.array(x_train)
	y_train = np.array(y_train)

	# y_train = sequence(y_train)
	# print(x_train.shape)
	embedding_size=32
	model=Sequential()
	model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))

	print(model.summary())
	model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])
	batch_size = 5
	num_epochs = 1
	x_valid, y_valid = x_train[:batch_size], y_train[:batch_size]
	x_train2, y_train2 = x_train[batch_size:], y_train[batch_size:]

	print(x_train2.shape)
	print(y_train2.shape)
	
	model.fit(x_train2, y_train2, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
	# scores = model.evaluate(steps=1)
			
			# word_list.append(words[0])
			# if words[1] in root then number is vocabulary_size-2
			# if words[1] in rel then number is vocabulary_size-1
			# if words[0] in dictionary, then number
			# else then position for UNK.
# 1) pull sentences with both concept types
# 2) figure out how to get word sequences within range that still will include both conceptids

# vector [1-50k, one for UNK, one for key, one for value]

# load_word_counts_dict()

# build()
serialize_data()
