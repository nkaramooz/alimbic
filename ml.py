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
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten
from keras.models import load_model
import re
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from collections import Counter
import string
import snomed_annotator as ann
from numpy import array
from numpy import asarray
from keras.layers import GlobalMaxPooling1D
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import multiprocessing
from gensim.models import Word2Vec
import sqlalchemy as sqla

plt.style.use('ggplot')

vocabulary_size = 20000

# need one for root (condition), need one for rel treatment
# need one for none (spacer)
vocabulary_spacer = 4
target_condition_key = vocabulary_size-3
generic_condition_key = vocabulary_size-4
target_treatment_key = vocabulary_size-2
max_words = 60
# [root, rel, spacer]

def get_word_counts():
	query = "select sentence from annotation.sentences4"
	conn,cursor = pg.return_postgres_cursor()
	sentence_df = pg.return_df_from_query(cursor, query, None, ['sentence'])
	
	counts = {}
	for i,s in sentence_df.iterrows():
		s = s['sentence']
		s = s.lower()
		s = ann.clean_text(s)
		words = s.split()

		for i,word in enumerate(words):
			if word in counts.keys():
				counts[word] = counts[word]+1
			else:
				counts[word] = 1

	counts = collections.OrderedDict(sorted(counts.items(), key=itemgetter(1), reverse = True))

	with open('word_count.pickle', 'wb') as handle:
		pk.dump(counts, handle, protocol=pk.HIGHEST_PROTOCOL)

	u.pprint(counts)
	return counts

def get_cid_and_word_counts():
	query = "select sentence_tuples from annotation.distinct_sentences"
	conn,cursor = pg.return_postgres_cursor()
	sentence_tuples_df = pg.return_df_from_query(cursor, query, None, ['sentence_tuples'])
	
	counts = {}
	for index,jb in sentence_tuples_df.iterrows():
		last_cid = ""
		for index,words in enumerate(jb[0]):
			if words[1] == last_cid:
				continue
			elif words[1] != 0:
				last_cid = words[1]
				if words[1] in counts.keys():
					counts[words[1]] = counts[words[1]] + 1
				else:
					counts[words[1]] = 1
			else:
				last_cid = ""
				if words[0] in counts.keys():
					counts[words[0]] = counts[words[0]] + 1
				else:
					counts[words[0]] = 1
	counts = collections.OrderedDict(sorted(counts.items(), key=itemgetter(1), reverse=True))

	with open('cid_and_word_count.pickle', 'wb') as handle:
		pk.dump(counts, handle, protocol=pk.HIGHEST_PROTOCOL)
	u.pprint(counts)
	return counts
	


def load_word_counts_dict(filename):
	with open('word_count.pickle', 'rb') as handle:
		counts = pk.load(handle)
		counter = 1
		final_dict = {}
		reverse_dict = {}
		final_dict['UNK'] = 0
		reverse_dict[0] = 'UNK'
		for item in counts.items():
			final_dict[item[0]] = counter
			reverse_dict[counter] = item[0]
			counter += 1
			if counter == (vocabulary_size-vocabulary_spacer):
				break

		with open('reversed_dictionary.pickle', 'wb') as rd:
			pk.dump(reverse_dict, rd, protocol=pk.HIGHEST_PROTOCOL)

		with open('dictionary.pickle', 'wb') as di:
			pk.dump(final_dict, di, protocol=pk.HIGHEST_PROTOCOL)
		# print(final_dict)
		# print(reverse_dict)
		# print(reverse_dict[1])


def gen_datasets_2(filename):

	conn,cursor = pg.return_postgres_cursor()
	labelled_query = "select condition_id, treatment_id, label from annotation.labelled_treatments order by condition_id"
	labelled_ids = pg.return_df_from_query(cursor, labelled_query, None, ["condition_id", "treatment_id", "label"])

	conditions_query = "select root_cid from annotation.concept_types where rel_type='condition' "
	all_conditions_set = set(pg.return_df_from_query(cursor, conditions_query, None, ["root_cid"])["root_cid"].tolist())

	dictionary, reverse_dictionary = get_dictionaries()

	all_sentences_df = pd.DataFrame()
	condition_id_cache = {}
	treatment_id_cache = {}

	condition_id_arr = []
	last_condition_id = ""
	condition_sentence_ids = pd.DataFrame()
	sentences_df = pd.DataFrame()
	for index,item in labelled_ids.iterrows():
		u.pprint(index)
		if item['condition_id'] == '%':

			tx_id_arr = None
			if item['treatment_id'] in treatment_id_cache.keys():
				tx_id_arr = treatment_id_cache[item['treatment_id']]
			else:
				tx_id_arr = [item['treatment_id']]
				tx_id_arr.extend(ann.get_children(item['treatment_id'], cursor))
				treatment_id_cache[item['treatment_id']] = tx_id_arr

			tx_sentence_id_query = "select id from annotation.sentences4 where conceptid in %s"
			tx_sentence_ids = pg.return_df_from_query(cursor, tx_sentence_id_query, (tuple(tx_id_arr),), ["id"])

			sentences_query = "select sentence_tuples, sentence, conceptid from annotation.sentences4 where id in %s and conceptid in %s"
			sentences_df = pg.return_df_from_query(cursor, sentences_query, (tuple(tx_sentence_ids), tuple(all_conditions_set)), ["sentence_tuples", "sentence", "conceptid"])

		else:
			if item['condition_id'] != last_condition_id:

				if item['condition_id'] in condition_id_cache.keys():
					condition_id_arr = condition_id_cache[item['condition_id']]
				else:

					condition_id_arr = [item['condition_id']]
					condition_id_arr.extend(ann.get_children(item['condition_id'], cursor))
					condition_id_cache[item['condition_id']] = condition_id_arr

				

				condition_sentence_id_query = "select id from annotation.sentences4 where conceptid in %s"
				condition_sentence_ids = pg.return_df_from_query(cursor, condition_sentence_id_query, (tuple(condition_id_arr),), ["id"])


			tx_id_arr = None
			if item['treatment_id'] in treatment_id_cache.keys():
				tx_id_arr = treatment_id_cache[item['treatment_id']]
			else:
				tx_id_arr = [item['treatment_id']]
				tx_id_arr.extend(ann.get_children(item['treatment_id'], cursor))
				treatment_id_cache[item['treatment_id']] = tx_id_arr

		#this should give a full list for the training row


			sentences_query = "select sentence_tuples, sentence, conceptid from annotation.sentences4 where id in %s and conceptid in %s"
			sentences_df = pg.return_df_from_query(cursor, sentences_query, (tuple(condition_sentence_ids['id'].tolist()), tuple(tx_id_arr)), ["sentence_tuples", "sentence", "conceptid"])
		last_condition_id = item['condition_id']
		label = item['label']

		all_sentences_df = all_sentences_df.append(get_labelled_data(True, sentences_df, condition_id_arr, tx_id_arr, item, dictionary, reverse_dictionary, all_conditions_set), sort=False)

	all_sentences_df = all_sentences_df.sample(frac=1).reset_index(drop=True)
	all_sentences_filename = "./all_sentences_" + str(filename)
	all_sentences_df.to_pickle(all_sentences_filename)
	all_sentences_df['rand'] = np.random.uniform(0,1, len(all_sentences_df))
	training_set = all_sentences_df[all_sentences_df['rand'] < 0.90].copy()
	testing_set = all_sentences_df[all_sentences_df['rand'] >= 0.90].copy()

	training_filename = "./training_" + str(filename)
	testing_filename = "./testing_" + str(filename)

	training_set.to_pickle(training_filename)
	testing_set.to_pickle(testing_filename)

	return training_filename, testing_filename


def get_sentences_df(condition_id_arr, tx_id_arr, cursor):

	sentence_query = "select id, sentence_tuples::text[], sentence, conceptid from annotation.sentences4 where conceptid in %s"
	sentence_df = pg.return_df_from_query(cursor, sentence_query, (tuple(condition_id_arr),), ["id", "sentence_tuples", "sentence", "conceptid"])
	treatments_df = "select id, sentence_tuples::text[], sentence, conceptid from annotation.sentences4 where conceptid in %s"
	treatments_df = pg.return_df_from_query(cursor, sentence_query, (tuple(tx_id_arr),), ["id", "sentence_tuples", "sentence", "conceptid"])

	sentence_df = sentence_df[sentence_df['id'].isin(treatments_df['id'])]
	
	return sentence_df

def get_dictionaries():
	with open('reversed_dictionary.pickle', 'rb') as rd:
  		reverse_dictionary = pk.load(rd)

	with open('dictionary.pickle', 'rb') as d:
		dictionary = pk.load(d)

	return dictionary, reverse_dictionary

def get_labelled_data_from_files(ids_filename, output_filename):
	conn,cursor = pg.return_postgres_cursor()
	labelled_ids = pd.read_pickle(ids_filename)
	print("labelled set length: " + str(len(labelled_ids)))

	dictionary, reverse_dictionary = get_dictionaries()

	conditions_query = "select root_cid from annotation.concept_types where rel_type='condition' "
	conditions_df = pg.return_df_from_query(cursor, conditions_query, None, ["root_cid"])
	conditions_df.columns = ['condition_id']

	results = pd.DataFrame()
	for index,item in labelled_ids.iterrows():
		u.pprint(index)
		condition_id_arr = [item['condition_id']]
		condition_id_arr.extend(ann.get_children(item['condition_id'], cursor))

		tx_id_arr = [item['treatment_id']]
		tx_id_arr.extend(ann.get_children(item['treatment_id'], cursor))

		#this should give a full list for the training row
		sentences_df = get_sentences_df(condition_id_arr, tx_id_arr, cursor)

		# Now you get the sentence, but you need to remove root and replace with word index
		# Create sentence list
		# each sentence list contains a list vector for word index
		label = item['label']

		results = results.append(get_labelled_data(True, sentences_df, condition_id_arr, tx_id_arr, item, dictionary, reverse_dictionary, conditions_df), sort=False)

	results.to_pickle(output_filename)

def get_labelled_data(is_test, sentences_df, condition_id_arr, tx_id_arr, item, dictionary, reverse_dictionary, conditions_set):
	final_results = pd.DataFrame()

	for index,sentence in sentences_df.iterrows():
		sample = [(vocabulary_size-1)]*max_words
		
		counter=0
		for index,words in enumerate(sentence[0]):
			## words[1] is the conceptid, words[0] is the word
			# condition 1 -- word is the condition of interest


			if (words[1] in condition_id_arr) and (sample[counter-1] != vocabulary_size-3):
				sample[counter] = vocabulary_size-3
				counter += 1
			# - word is treatment of interest
			elif words[1] not in condition_id_arr and words[1] in conditions_set and (sample[counter-1] != generic_condition_key):
				sample[counter] = str(generic_condition_key)
				counter += 1
			elif (words[1] in tx_id_arr) and (sample[counter-1] != vocabulary_size-2):
				sample[counter] = vocabulary_size-2
				counter += 1
			# other condition
			elif words[1] in conditions_set and (sample[counter-1] != vocabulary_size-4) and (words[1] not in condition_id_arr):
				sample[counter] = vocabulary_size-4
				counter += 1
			# Now using conceptid if available
			elif words[1] != '0' and words[1] in dictionary.keys() and dictionary[words[1]] != sample[counter-1]:
				sample[counter] = dictionary[words[1]]
				counter += 1
			elif words[0] in dictionary.keys() and dictionary[words[0]] != sample[counter-1] and words[1] not in condition_id_arr and words[1] not in tx_id_arr:
				sample[counter] = dictionary[words[0]]
				counter += 1
			elif words[0] not in dictionary.keys() and words[1] not in condition_id_arr and words[1] not in tx_id_arr:
				sample[counter] = dictionary['UNK']
				counter += 1
			if counter >= max_words-1:
				break
				# while going through, see if conceptid associated then add to dataframe with root index and rel_type index
		if is_test:
			final_results = final_results.append(pd.DataFrame([[sentence['sentence'],sentence['sentence_tuples'], item['condition_id'], item['treatment_id'], sample, item['label']]], columns=['sentence', 'sentence_tuples', 'condition_id', 'treatment_id', 'x_train', 'label']))
		else:
			final_results = final_results.append(pd.DataFrame([[sentence['sentence'],sentence['sentence_tuples'], item['condition_id'], item['treatment_id'], sample, sentence['pmid'], sentence['id']]], columns=['sentence', 'sentence_tuples', 'condition_id', 'treatment_id', 'x_train', 'pmid', 'id']))

	return final_results

def get_labelled_data_w2v(sentences_df, conditions_set, dictionary, reverse_dictionary):
	final_results = []

	for index,sentence in sentences_df.iterrows():
		sample = [str((vocabulary_size-1))]*max_words
		
		counter=0
		for index,words in enumerate(sentence[0]):
			## words[1] is the conceptid, words[0] is the word
			# condition 1 -- word is the condition of interest
	
			if words[1] in conditions_set and (sample[counter-1] != generic_condition_key):
				sample[counter] = str(generic_condition_key)
				counter += 1
			# Now using conceptid if available
			elif words[1] != '0' and words[1] in dictionary.keys() and dictionary[words[1]] != sample[counter-1]:
				sample[counter] = str(dictionary[words[1]])
				counter += 1
			elif words[0] in dictionary.keys() and dictionary[words[0]] != sample[counter-1]:
				sample[counter] = str(dictionary[words[0]])
				counter += 1
			elif words[0] not in dictionary.keys():
				sample[counter] = str(dictionary['UNK'])
				counter += 1
			if counter >= max_words-1:
				break
				# while going through, see if conceptid associated then add to dataframe with root index and rel_type index

		final_results.append(sample)

	return final_results


def build_embedding():
	dictionary, reverse_dictionary = get_dictionaries()
	conn,cursor = pg.return_postgres_cursor()
	conditions_query = "select root_cid from annotation.concept_types where rel_type='condition' "
	all_conditions_set = set(pg.return_df_from_query(cursor, conditions_query, None, ["root_cid"])["root_cid"].tolist())

	max_len_query = "select count(*) as cnt from annotation.distinct_sentences"
	max_len = pg.return_df_from_query(cursor, max_len_query, None, ['cnt'])['cnt'].values
	
	counter = 0
	
	model = None
	while counter < max_len:
		sentence_vector = []
		u.pprint(counter)
		sentences_query = "select sentence_tuples from annotation.distinct_sentences limit 100000 offset %s"
		sentences_df = pg.return_df_from_query(cursor, sentences_query, (counter,), ['sentence_tuples'])	
		sentence_vector = get_labelled_data_w2v(sentences_df, all_conditions_set, dictionary, reverse_dictionary)

		if model == None:
			model = Word2Vec(sentence_vector, size=200, window=5, min_count=1, negative=15, iter=10, workers=multiprocessing.cpu_count())
		else:
			model.build_vocab(sentence_vector, update=True)
			model.train(sentence_vector, total_examples=len(sentence_vector), epochs=3)
		counter += 100000

	model.save('concept_word_embedding.200.04.21.bin')


def confirmation(filename):
	conn,cursor = pg.return_postgres_cursor()
	labelled_query = "select condition_id, treatment_id, label from annotation.labelled_treatments_confirmation"
	labelled_ids = pg.return_df_from_query(cursor, labelled_query, None, ["condition_id", "treatment_id", "label"])
	labelled_ids.to_pickle("./confirmation_ids")

	get_labelled_data_from_files("./confirmation_ids", "./confirmation_60.pkl")

	test_set = pd.read_pickle("./confirmation_60.pkl")

	x_test = np.array(test_set['x_train'].tolist())
	y_test = np.array(test_set['label'].tolist())
	model = load_model(filename)

	correct_counter = 0

	for i,d in test_set.iterrows():
		
		prediction = model.predict(np.array([d['x_train']]))
		
		if ((prediction > 0.50) and (d['label'] == 0)) or ((prediction <= 0.50) and (d['label'] == 1)):
			print(d['label'])
			print(prediction)
			print(d['sentence'])
			print(d['x_train'])
			print("condition_id : " + str(d['condition_id']) + "treatment_id : " + str(d['treatment_id']))
			continue 
		else:
			print(d['label'])
			print(prediction)
			print(d['sentence'])
			print(d['x_train'])
			print("condition_id : " + str(d['condition_id']) + "treatment_id : " + str(d['treatment_id']))

			print("=========" + str(i))
			correct_counter += 1

	print(correct_counter)
	print(Counter(test_set['label'].tolist()))


def treatment_recategorization_recs(model_name):
	
	conn,cursor = pg.return_postgres_cursor()
	dictionary, reverse_dictionary = get_dictionaries()
	model = load_model(model_name)

	conditions_query = "select root_cid from annotation.concept_types where rel_type='condition' "
	all_conditions_set = set(pg.return_df_from_query(cursor, conditions_query, None, ["root_cid"])["root_cid"].tolist())

	max_query = "select count(*) as cnt from annotation.title_treatment_candidates_filtered"
	max_counter = pg.return_df_from_query(cursor, max_query, None, ['cnt'])['cnt'].values
	print(max_counter)
	counter = 0

	while counter < max_counter:
		print(counter)
		results_df = pd.DataFrame()
		treatment_candidates_query = "select sentence, sentence_tuples, condition_id, treatment_id, pmid, id, section from annotation.title_treatment_candidates_filtered limit 1000 offset %s"
		treatment_candidates_df = pg.return_df_from_query(cursor, treatment_candidates_query, (counter,), ['sentence', 'sentence_tuples', 'condition_id', 'treatment_id', 'pmid', 'id', 'section'])

		for i,c in treatment_candidates_df.iterrows():
			row_df = get_labelled_data_sentence(c, c['condition_id'], c['treatment_id'], dictionary, reverse_dictionary, all_conditions_set)
		
			row_df['score'] = model.predict(np.array([row_df['x_train'].values[0]]))[0][0]

			results_df = results_df.append(row_df[['sentence', 'sentence_tuples', 'condition_id', 'treatment_id', 'score', 'pmid', 'id']])

		if counter == 0:
			engine = pg.return_sql_alchemy_engine()
			results_df.to_sql('raw_treatment_recs_staging', engine, schema='annotation', if_exists='replace', index=False, dtype={'sentence_tuples' : sqla.types.JSON, 'concept_arr' : sqla.types.JSON})
		else:
			engine = pg.return_sql_alchemy_engine()
			results_df.to_sql('raw_treatment_recs_staging', engine, schema='annotation', if_exists='append', index=False, dtype={'sentence_tuples' : sqla.types.JSON, 'concept_arr' : sqla.types.JSON})

		counter += 1000
	# u.pprint(results_df)
	cursor.close()

def get_labelled_data_sentence(sentence, condition_id, tx_id, dictionary, reverse_dictionary, conditions_set):
	final_results = pd.DataFrame()

	sample = [(vocabulary_size-1)]*max_words
	
	counter=0
	for index,words in enumerate(sentence['sentence_tuples']):
		## words[1] is the conceptid, words[0] is the word
		# condition 1 -- word is the condition of interest
		if (words[1] == condition_id) and (sample[counter-1] != vocabulary_size-3):
			sample[counter] = vocabulary_size-3
			counter += 1
		# - word is treatment of interest
		elif words[1] != condition_id and words[1] in conditions_set and (sample[counter-1] != generic_condition_key):
			sample[counter] = str(generic_condition_key)
			counter += 1
		elif (words[1] == tx_id) and (sample[counter-1] != vocabulary_size-2):
			sample[counter] = vocabulary_size-2
			counter += 1
		# other condition
		elif words[1] in conditions_set and (sample[counter-1] != vocabulary_size-4) and (words[1] != condition_id):
			sample[counter] = vocabulary_size-4
			counter += 1
		# Now using conceptid if available
		elif words[1] != '0' and words[1] in dictionary.keys() and dictionary[words[1]] != sample[counter-1]:
			sample[counter] = dictionary[words[1]]
			counter += 1
		elif words[0] in dictionary.keys() and dictionary[words[0]] != sample[counter-1] and words[1] != condition_id and words[1] != tx_id:
			sample[counter] = dictionary[words[0]]
			counter += 1
		elif words[0] not in dictionary.keys() and words[1] != condition_id and words[1] != tx_id:
			sample[counter] = dictionary['UNK']
			counter += 1
		if counter >= max_words-1:
			break
			# while going through, see if conceptid associated then add to dataframe with root index and rel_type index
	return pd.DataFrame([[sentence['sentence'], sentence['sentence_tuples'], condition_id, tx_id, sample, sentence['pmid'], sentence['id']]], columns=['sentence', 'sentence_tuples', 'condition_id', 'treatment_id', 'x_train', 'pmid', 'id'])

	

def train_with_word2vec():
	model = Word2Vec.load('concept_word_embedding.200.04.21.bin')
	
	embedding_dim = 200
	embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
	dictionary, reverse_dictionary = get_dictionaries()

	counter = 0
	for key in dictionary:
		embedding_vector = None
		if key in model.wv.vocab:
			embedding_vector = model[key]

		if embedding_vector is not None:
			embedding_matrix[dictionary[key]] = embedding_vector
		counter += 1
		if counter >= (vocabulary_size-vocabulary_spacer):
			break

	training_set = pd.read_pickle("./training_05_18_19")
	training_set = training_set.sample(frac=1).reset_index(drop=True)
	test_set = pd.read_pickle("./testing_05_18_19")

	x_train = np.array(training_set['x_train'].tolist())
	y_train = np.array(training_set['label'].tolist())

	x_test = np.array(test_set['x_train'].tolist())
	y_test = np.array(test_set['label'].tolist())
	
	embedding_size=200
	batch_size = 128
	num_epochs = 4
	model=Sequential()
	model.add(Embedding(vocabulary_size, embedding_size, weights=[embedding_matrix], input_length=max_words, trainable=True))
	model.add(LSTM(800, return_sequences=True, input_shape=(embedding_size, batch_size)))
	model.add(Dropout(0.3))
	model.add(LSTM(800, return_sequences=True))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))

	print(model.summary())
	model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

	history = model.fit(x_train, y_train, validation_split = 0.1, batch_size=batch_size, epochs=num_epochs, shuffle='batch')


	loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
	print("Training Accuracy: {:.4f}".format(accuracy))
	loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
	print("Testing Accuracy:  {:.4f}".format(accuracy))
	# plot_history(history)

	model.save('txp_60_05_18_w2v.h5')	


def plot_history(history):
    acc = history.history['acc']
    u.pprint(acc)
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()



# build_model()


# get_cid_and_word_counts()
# load_word_counts_dict("cid_and_word_count.pickle")
# build_embedding()
gen_datasets_2("05_18_19")
train_with_word2vec()
# treatment_recategorization_recs('txp_60_05_17_w2v.h5')
# confirmation('txp_60_03_02_w2v.h5')


