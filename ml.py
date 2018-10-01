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
labeled_set = pd.DataFrame([[]], columns=['root', 'rel', 'label'])

def get_data():
	conn,cursor = pg.return_postgres_cursor()
	labelled_set = pd.DataFrame()
	for index,item in labeled_set.iterrows():
		query = "select subtypeid from snomed.transitive_closure where supertypeid=%s"
		# Run this twice, once for root and once for rel_type

		# Want to select sentenceIds that contain both
		next_query = "select sentence_tuples from annotation.sentences where conceptid in %s and \
			sentence_id in (select sentence_id from annotation.sentences where conceptid in %s) limit 1"

		#this should give a full list for the training row
		sentence = pg.return_df_from_query(cursor, new_candidate_query, (word, word), [""])
		# Now you get the sentence, but you need to remove root and replace with word index
		# Create sentence list
		# each sentence list contains a list vector for word index
		for words in s['sentence_tuples']:
			words = words.strip('(')
			words = words.strip(')')
			words = tuple(words.split(","))
			
			# word_list.append(words[0])
			# if words[1] in root then number is vocabulary_size-2
			# if words[1] in rel then number is vocabulary_size-1
			# if words[0] in dictionary, then number
			# else then position for UNK.
# 1) pull sentences with both concept types
# 2) figure out how to get word sequences within range that still will include both conceptids

# vector [1-50k, one for UNK, one for key, one for value]

# load_word_counts_dict()