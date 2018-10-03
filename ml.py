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
  	pickle.dump(reversed_dictionary, rd, protocol=pickle.HIGHEST_PROTOCOL)

  with open('dictionary.pickle', 'wb') as di:
  	pickle.dump(dictionary, di, protocol=pickle.HIGHEST_PROTOCOL)

  return data, count, dictionary, reversed_dictionary


def build():

	vocabulary = read_data()
	vocabulary_size = 50000

	data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
	                                                            vocabulary_size)
	del vocabulary

	print(count)

def load_word_counts_dict():
	with open('word_count.pickle', 'rb') as handle:
		counts = pickle.load(handle)
		print(counts[0:50000])

# How does training 
#[[X (i.e. condition), Y (i.e. treatment), label]]
# Will need to get all children of X, and all children of Y
labeled_set = [['22298006','387458008', '1']]

def get_data():
	conn,cursor = pg.return_postgres_cursor()
	labelled_set = pd.DataFrame()
	for index,item in enumerate(labeled_set):
		
		root_cids = [item[0]]
		root_cids.extend(ann.get_children(item[0], cursor))

		rel_cids = [item[1]]
		rel_cids.extend(ann.get_children(item[1], cursor))


		# Want to select sentenceIds that contain both
		sentence_query = "select sentence_tuples from annotation.sentences where conceptid in %s and id in (select id from annotation.sentences where conceptid in %s) limit 10"

		#this should give a full list for the training row
		sentences = pg.return_df_from_query(cursor, sentence_query, (tuple(root_cids), tuple(rel_cids)), ["sentence_tuples"])
		# Now you get the sentence, but you need to remove root and replace with word index
		# Create sentence list
		# each sentence list contains a list vector for word index
		results = pd.DataFrame()
	
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
			results = results.append(pd.DataFrame([[sentence, sentence_root_index, sentence_rel_index]], columns=['sentence_tuples', 'root_index', 'rel_index']))
		
		u.pprint(results)
		break
			
			# word_list.append(words[0])
			# if words[1] in root then number is vocabulary_size-2
			# if words[1] in rel then number is vocabulary_size-1
			# if words[0] in dictionary, then number
			# else then position for UNK.
# 1) pull sentences with both concept types
# 2) figure out how to get word sequences within range that still will include both conceptids

# vector [1-50k, one for UNK, one for key, one for value]

# load_word_counts_dict()

get_data()