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
  return data, count, dictionary, reversed_dictionary

# with open('reversed_dictionary.pickle', 'wb') as rd:
#   pickle.dump(reversed_dictionary, rd, protocol=pickle.HIGHEST_PROTOCOL)

# with open('dictionary.pickle', 'wb') as di:
#   pickle.dump(dictionary, di, protocol=pickle.HIGHEST_PROTOCOL)

vocabulary = read_data()
vocabulary_size = 50000

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
print(count)

def load_word_counts_dict():
	with open('word_count.pickle', 'rb') as handle:
		counts = pickle.load(handle)
		print(counts[0:50000])


# vector [1-50k, one for UNK, one for key, one for value]

# load_word_counts_dict()