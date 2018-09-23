import pandas as pd
import re
import pickle
import psycopg2
from sqlalchemy import create_engine
from nltk.stem.wordnet import WordNetLemmatizer
from collections import OrderedDict
from operator import itemgetter
import nltk.data
import numpy as np
import time
import utilities.utils as u, utilities.pglib as pg





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
		pickle.dump(counts, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return counts

def load_word_counts_dict():
	with open('word_count.pickle', 'rb') as handle:
		counts = pickle.load(handle)
		u.pprint(len(counts))


# vector [1-50k, one for UNK, one for key, one for value]

load_word_counts_dict()