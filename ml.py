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


plt.style.use('ggplot')

vocabulary_size = 10000

# need one for root (condition), need one for rel treatment
# need one for none (spacer)
vocabulary_spacer = 4
target_condition_key = vocabulary_size-3
generic_condition_key = vocabulary_size-4
target_treatment_key = vocabulary_size-2
max_words = 40
# [root, rel, spacer]

def get_word_counts():
	query = "select sentence from annotation.sentences3"
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
	query = "select distinct(sentence_tuples)::text[] from annotation.sentences3"
	conn,cursor = pg.return_postgres_cursor()
	sentence_tuples_df = pg.return_df_from_query(cursor, query, None, ['sentence_tuples'])
	
	counts = {}
	for index,sentence in sentence_tuples_df.iterrows():
		last_cid = ""
		for index,words in enumerate(sentence['sentence_tuples']):
		
			words = words.lower()
			words = words.strip('(')
			words = words.strip(')')
			words = tuple(words.split(","))	

			if (words[1] == last_cid):
				continue
			elif (words[1] != '0'):
				last_cid = words[1]
				if words[1] in counts.keys():
					counts[words[1]] = counts[words[1]] + 1
				else:
					counts[words[1]] = 1
			else:
				last_cid = ""
				clean_word = ann.clean_text(words[0])
				if clean_word in counts.keys():
					counts[clean_word] = counts[clean_word] + 1
				else:
					counts[clean_word] = 1
	counts = collections.OrderedDict(sorted(counts.items(), key=itemgetter(1), reverse = True))
	
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


# This function will split the data and generate the files
def gen_datasets():
	conn,cursor = pg.return_postgres_cursor()
	labelled_query = "select condition_id, treatment_id, label from annotation.labelled_treatments"
	labelled_ids = pg.return_df_from_query(cursor, labelled_query, None, ["condition_id", "treatment_id", "label"])
	labelled_ids['rand'] = np.random.uniform(0, 1, len(labelled_ids))

	training_set = labelled_ids[labelled_ids['rand'] <= 0.90].copy()
	testing_set = labelled_ids[labelled_ids['rand'] > 0.90].copy()

	print("training_set length: " + str(len(training_set)))
	print("testing_set length: " + str(len(testing_set)))

	training_set.to_pickle("./training_ids_02_08_19")
	testing_set.to_pickle("./testing_ids_02_08_19")

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
	for index,item in labelled_ids.iterrows():
		u.pprint(index)
		if item['condition_id'] != last_condition_id:

			if item['condition_id'] in condition_id_cache.keys():
				condition_id_arr = condition_id_cache[item['condition_id']]
			else:

				condition_id_arr = [item['condition_id']]
				condition_id_arr.extend(ann.get_children(item['condition_id'], cursor))
				condition_id_cache[item['condition_id']] = condition_id_arr

			last_condition_id = item['condition_id']

			condition_sentence_id_query = "select id from annotation.sentences3 where conceptid in %s"
			condition_sentence_ids = pg.return_df_from_query(cursor, condition_sentence_id_query, (tuple(condition_id_arr),), ["id"])


		tx_id_arr = None
		if item['treatment_id'] in treatment_id_cache.keys():
			tx_id_arr = treatment_id_cache[item['treatment_id']]
		else:
			tx_id_arr = [item['treatment_id']]
			tx_id_arr.extend(ann.get_children(item['treatment_id'], cursor))
			treatment_id_cache[item['treatment_id']] = tx_id_arr


		#this should give a full list for the training row


		sentences_query = "select sentence_tuples::text[], sentence, conceptid from annotation.sentences3 where id in %s and conceptid in %s"
		sentences_df = pg.return_df_from_query(cursor, sentences_query, (tuple(condition_sentence_ids['id'].tolist()), tuple(tx_id_arr)), ["sentence_tuples", "sentence", "conceptid"])

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

def gen_datasets_w2v(filename):

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
	for index,item in labelled_ids.iterrows():
		u.pprint(index)
		if item['condition_id'] != last_condition_id:

			if item['condition_id'] in condition_id_cache.keys():
				condition_id_arr = condition_id_cache[item['condition_id']]
			else:

				condition_id_arr = [item['condition_id']]
				condition_id_arr.extend(ann.get_children(item['condition_id'], cursor))
				condition_id_cache[item['condition_id']] = condition_id_arr

			last_condition_id = item['condition_id']

			condition_sentence_id_query = "select id from annotation.sentences3 where conceptid in %s"
			condition_sentence_ids = pg.return_df_from_query(cursor, condition_sentence_id_query, (tuple(condition_id_arr),), ["id"])


		tx_id_arr = None
		if item['treatment_id'] in treatment_id_cache.keys():
			tx_id_arr = treatment_id_cache[item['treatment_id']]
		else:
			tx_id_arr = [item['treatment_id']]
			tx_id_arr.extend(ann.get_children(item['treatment_id'], cursor))
			treatment_id_cache[item['treatment_id']] = tx_id_arr

		sentences_query = "select sentence_tuples::text[], sentence, conceptid from annotation.sentences3 where id in %s and conceptid in %s"
		sentences_df = pg.return_df_from_query(cursor, sentences_query, (tuple(condition_sentence_ids['id'].tolist()), tuple(tx_id_arr)), ["sentence_tuples", "sentence", "conceptid"])

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

	sentence_query = "select id, sentence_tuples::text[], sentence, conceptid from annotation.sentences3 where conceptid in %s"
	sentence_df = pg.return_df_from_query(cursor, sentence_query, (tuple(condition_id_arr),), ["id", "sentence_tuples", "sentence", "conceptid"])
	treatments_df = "select id, sentence_tuples::text[], sentence, conceptid from annotation.sentences3 where conceptid in %s"
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
		for index,words in enumerate(sentence['sentence_tuples']):
			# words = words.strip('{')
			# words = words.strip('}')
			words = words.lower()
			words = words.strip('(')
			words = words.strip(')')
			words = tuple(words.split(","))	

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
		for index,words in enumerate(sentence['sentence_tuples']):
			# words = words.strip('{')
			# words = words.strip('}')
			words = words.lower()
			words = words.strip('(')
			words = words.strip(')')
			words = tuple(words.split(","))	

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

	max_len = 12990566
	counter = 0
	
	model = None
	while counter < max_len:
		sentence_vector = []
		u.pprint(counter)
		sentences_query = "select sentence_tuples::text[] from annotation.sentences3 limit 100000 offset %s"
		sentences_df = pg.return_df_from_query(cursor, sentences_query, (counter,), ['sentence_tuples'])	
		sentence_vector = get_labelled_data_w2v(sentences_df, all_conditions_set, dictionary, reverse_dictionary)

		if model == None:
			model = Word2Vec(sentence_vector, size=200, window=5, min_count=1, negative=15, iter=10, workers=multiprocessing.cpu_count())
		else:
			model.build_vocab(sentence_vector, update=True)
			model.train(sentence_vector, total_examples=len(sentence_vector), epochs=3)
		counter += 100000

		# for index,sentence in all_sentences_df.iterrows():
		# 	sentence_array = []
		# 	for index,words in enumerate(sentence['sentence_tuples']):

		# 		words = words.lower()
		# 		words = words.strip('(')
		# 		words = words.strip(')')
		# 		words = tuple(words.split(","))	

		# 		## words[1] is the conceptid, words[0] is the word

		# 		# condition 1 -- word is the condition of interest

		# 		# if words[1] != '0' and words[1] in condition_array:
		# 		# 	sentence_array.append(str(generic_condition_key))
		# 		if words[1] != '0':
		# 			sentence_array.append(str(words[1]))
		# 		else:
		# 			sentence_array.append(str(words[0]))

		# 	final_array.append(sentence_array)

	model.save('concept_word_embedding.200.bin')


def build_model():
	# pd.set_option('display.max_colwidth', -1)
	training_set = pd.read_pickle("./training_02_23_19.2")
	training_set = training_set.append(training_set[training_set['label'] == 1].copy())
	training_set = training_set.sample(frac=1).reset_index(drop=True)

	test_set = pd.read_pickle("./testing_02_23_19.2")

	x_train = np.array(training_set['x_train'].tolist())
	y_train = np.array(training_set['label'].tolist())

	x_test = np.array(test_set['x_train'].tolist())
	y_test = np.array(test_set['label'].tolist())
	

	# param_grid = dict(num_filters=[32, 64, 128],
	# 	kernel_size = [3,7,10, 20],
	# 	vocab_size=vocabulary_size,
	# 	embedding_dim=[100, 200, 500],
	# 	maxlen=[max_words])
	# model = KerasClassifier(build_fn=create_model, epochs=num_epochs, batch_size=batch_size, verbose=False)
	# grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=4, verbose=1, n_iter=5)
	# grid_result = grid.fit(x_train, y_train)
	# test_accuracy = grid.score(x_test, y_test)

	# prompt = input(f'finished {source}; write to file and proceed? [y/n]')
	# if prompt.lower() not in ['y', 'true', 'yes']:
	# 	break
	# with open(output_file, 'a') as f:
	# 	s = ('Running {} data set\nBest Accuracy : '
	# 	'{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
	# 	output_string = s.format(
	# 		source,
	# 		grid_result.best_score_,
	# 		grid_result.best_params_,
	# 		test_accuracy)
	# 	print(output_string)
	# 	f.write(output_string)

	# model.add(Conv1D(filters=150, kernel_size=3, padding='same', activation='relu'))
	# model.add(GlobalMaxPooling1D())
	# model.add(MaxPooling1D(pool_size=2))
	# model.add(LSTM(100))
		# model.add(Dropout(0.2))
	# model.add(LSTM(100, return_sequences=True))
	# model.add(Conv1D(filters=100, kernel_size=10, padding='same', activation='relu'))
	# model.add(MaxPooling1D(pool_size=10))
	# model.add(LSTM(100, return_sequences=True))
	embedding_size=200
	batch_size = 100
	num_epochs = 4

	model=Sequential()
	model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(LSTM(100, return_sequences=True, input_shape=(embedding_size, batch_size)))
	model.add(LSTM(100, return_sequences=True))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))

	print(model.summary())
	model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

	history = model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=num_epochs, verbose=True, shuffle='batch')


	loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
	print("Training Accuracy: {:.4f}".format(accuracy))
	loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
	print("Testing Accuracy:  {:.4f}".format(accuracy))
	plot_history(history)
	model.save('txp_60_02_24.1.h5')

def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

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
	# pd.set_option('display.max_colwidth', -1)
	conn,cursor = pg.return_postgres_cursor()
	dictionary, reverse_dictionary = get_dictionaries()
	model = load_model(model_name)

	

	conditions_query = "select root_cid from annotation.concept_types where rel_type='condition' "
	all_conditions_set = set(pg.return_df_from_query(cursor, conditions_query, None, ["root_cid"])["root_cid"].tolist())

	

	counter = 0
	max_counter = 1348396
	while counter < max_counter:
		print(counter)
		results_df = pd.DataFrame()
		treatment_candidates_query = "select sentence, sentence_tuples::text[], condition_id, treatment_id, pmid, id, section from annotation.title_treatment_candidates_filtered limit 1000 offset %s"
		treatment_candidates_df = pg.return_df_from_query(cursor, treatment_candidates_query, (counter,), ['sentence', 'sentence_tuples', 'condition_id', 'treatment_id', 'pmid', 'id', 'section'])

		for i,c in treatment_candidates_df.iterrows():
			row_df = get_labelled_data_sentence(c, c['condition_id'], c['treatment_id'], dictionary, reverse_dictionary, all_conditions_set)
		
			row_df['score'] = model.predict(np.array([row_df['x_train'].values[0]]))[0][0]

			results_df = results_df.append(row_df[['sentence', 'sentence_tuples', 'condition_id', 'treatment_id', 'score', 'pmid', 'id']])

		if counter == 0:
			engine = pg.return_sql_alchemy_engine()
			results_df.to_sql('raw_treatment_recs', engine, schema='annotation', if_exists='replace', index=False)
		else:
			engine = pg.return_sql_alchemy_engine()
			results_df.to_sql('raw_treatment_recs', engine, schema='annotation', if_exists='append', index=False)

		counter += 1000
	# u.pprint(results_df)
	cursor.close()

def get_labelled_data_sentence(sentence, condition_id, tx_id, dictionary, reverse_dictionary, conditions_set):
	final_results = pd.DataFrame()


	sample = [(vocabulary_size-1)]*max_words
	
	counter=0
	for index,words in enumerate(sentence['sentence_tuples']):
		# words = words.strip('{')
		# words = words.strip('}')
		words = words.lower()
		words = words.strip('(')
		words = words.strip(')')
		words = tuple(words.split(","))	
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
	model = Word2Vec.load('concept_word_embedding.200.bin')
	
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

	training_set = pd.read_pickle("./training_03_14_19")
	# training_set = training_set.append(training_set[training_set['label'] == 1].copy())
	training_set = training_set.sample(frac=1).reset_index(drop=True)
	test_set = pd.read_pickle("./testing_03_14_19")

	x_train = np.array(training_set['x_train'].tolist())
	y_train = np.array(training_set['label'].tolist())

	x_test = np.array(test_set['x_train'].tolist())
	y_test = np.array(test_set['label'].tolist())
	
	embedding_size=200
	batch_size = 64
	num_epochs = 3
	model=Sequential()
	model.add(Embedding(vocabulary_size, embedding_size, weights=[embedding_matrix], input_length=max_words, trainable=True))
	# model.add(Conv1D(filters=10, kernel_size=3, padding='same', activation='relu'))
	# model.add(MaxPooling1D(pool_size=5))
	model.add(LSTM(800, return_sequences=True, input_shape=(embedding_size, batch_size)))
	model.add(Dropout(0.2))
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

	model.save('txp_60_03_14_w2v.h5')


def glove_embeddings():
	embeddings_index = dict()
	f = open('glove.6B.100d.txt')
	for line in f:
		values = line.split()
		word = values[0]
		coefs = asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	print("======= loaded glove ======")
	u.pprint(embeddings_index)

	embedding_matrix = np.zeros((vocabulary_size, 100))
	dictionary, reverse_dictionary = get_dictionaries()

	counter = 0
	for key in dictionary:
		embedding_vector = embeddings_index.get(key)
		if embedding_vector is not None:
			embedding_matrix[dictionary[key]] = embedding_vector
		counter += 1
		if counter >= (vocabulary_size-vocabulary_spacer):
			break

	print("======= completed embedding_matrix ======")
	u.pprint(len(embedding_matrix))
	u.pprint(embedding_matrix)
	training_set = pd.read_pickle("./training_02_16_19")
	training_set = training_set.append(training_set[training_set['label'] == 1].copy())
	training_set = training_set.sample(frac=1).reset_index(drop=True)
	test_set = pd.read_pickle("./testing_02_16_19")

	x_train = np.array(training_set['x_train'].tolist())
	y_train = np.array(training_set['label'].tolist())

	x_test = np.array(test_set['x_train'].tolist())
	y_test = np.array(test_set['label'].tolist())
	
	embedding_size=64
	batch_size = 32
	num_epochs = 2
	model=Sequential()
	model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
	model.add(LSTM(100, return_sequences=True, input_shape=(embedding_size, batch_size)))

	
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
	plot_history(history)

	model.save('txp_60_02_22_glove.h5')
	

	
	
	# model = load_model('txp_60_02_21_glove.h5')

	# correct_counter = 0
	# counter = 0
	# for i,d in test_set.iterrows():
		
	# 	prediction = model.predict(np.array([d['x_train']]))
		

	# 	if ((prediction > 0.50) and (d['label'] == 0)) or ((prediction <= 0.50) and (d['label'] == 1)):
	# 		print(d['label'])
	# 		print(prediction)
	# 		print(d['sentence'])
	# 		print(d['x_train'])
	# 		print("condition_id : " + str(d['condition_id']) + " treatment_id : " + str(d['treatment_id']))

	# 		counter+=1
	# 		print("=========" + str(i))
	# 	else:
	# 		correct_counter += 1
	# 	# if counter >= 100:
	# 	# 	break
		
	# print(correct_counter)
	# print(len(test_set[test_set['label'] == 1]))
	# print(Counter(test_set['label'].tolist()))

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

# glove_embeddings()

# build_model()



# build_embedding()
# get_cid_and_word_counts()
# load_word_counts_dict("cid_and_word_count.pickle")
# gen_datasets_2("03_14_19")

# train_with_word2vec()
treatment_recategorization_recs('txp_60_03_14_w2v.h5')
# confirmation('txp_60_03_02_w2v.h5')

# model = Word2Vec.load('concept_word_embedding.bin')
# print(model.wv.most_similar(positive=['29857009', '267036007']))

