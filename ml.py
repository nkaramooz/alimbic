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

vocabulary_size = 50000

# need one for root (condition), need one for rel treatment
# need one for none (spacer)
vocabulary_spacer = 3
max_words = 60
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


def load_word_counts_dict():
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

	training_set = labelled_ids[labelled_ids['rand'] <= 0.85].copy()
	testing_set = labelled_ids[labelled_ids['rand'] > 0.85].copy()
	print("training_set length: " + str(len(training_set)))
	print("testing_set length: " + str(len(testing_set)))

	training_set.to_pickle("./training_ids_02_08_19")
	testing_set.to_pickle("./testing_ids_02_08_19")

def gen_datasets_2(filename):
	conn,cursor = pg.return_postgres_cursor()
	labelled_query = "select condition_id, treatment_id, label from annotation.labelled_treatments"
	labelled_ids = pg.return_df_from_query(cursor, labelled_query, None, ["condition_id", "treatment_id", "label"])
	dictionary, reverse_dictionary = get_dictionaries()

	all_sentences_df = pd.DataFrame()
	condition_id_arr = []
	last_condition_id = ""
	for index,item in labelled_ids.iterrows():
		u.pprint(index)
		if item['condition_id'] != last_condition_id:
			condition_id_arr = [item['condition_id']]
			condition_id_arr.extend(ann.get_children(item['condition_id'], cursor))
			last_condition_id = item['condition_id']

		tx_id_arr = [item['treatment_id']]
		tx_id_arr.extend(ann.get_children(item['treatment_id'], cursor))

		#this should give a full list for the training row
		sentences_df = get_sentences_df(condition_id_arr, tx_id_arr, cursor)

		# Now you get the sentence, but you need to remove root and replace with word index
		# Create sentence list
		# each sentence list contains a list vector for word index
		label = item['label']

		all_sentences_df = all_sentences_df.append(get_labelled_data(True, sentences_df, condition_id_arr, tx_id_arr, item, dictionary, reverse_dictionary), sort=False)

	all_sentences_df.to_pickle("all_sentences.pk")
	all_sentences_df['rand'] = np.random.uniform(0,1, len(all_sentences_df))
	training_set = all_sentences_df[all_sentences_df['rand'] <= 0.85].copy()
	testing_set = all_sentences_df[all_sentences_df['rand'] > 0.85].copy()

	training_filename = "./training_" + str(filename)
	testing_filename = "./testing_" + str(filename)

	training_set.to_pickle(training_filename)
	testing_set.to_pickle(testing_filename)

	return training_filename, testing_filename
	

def get_sentences_df(condition_id_arr, tx_id_arr, cursor):
	sentence_query = "select sentence_tuples::text[], sentence from annotation.sentences3 where conceptid in %s \
		and id in (select id from annotation.sentences3 where conceptid in %s)"
	sentence_tuples = pg.return_df_from_query(cursor, sentence_query, (tuple(condition_id_arr), tuple(tx_id_arr)), ["sentence_tuples", "sentence"])
	return sentence_tuples

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

		results = results.append(get_labelled_data(True, sentences_df, condition_id_arr, tx_id_arr, item, dictionary, reverse_dictionary), sort=False)

	results.to_pickle(output_filename)

def get_labelled_data(is_test, sentences_df, condition_id_arr, tx_id_arr, item, dictionary, reverse_dictionary):
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

			if (words[1] in condition_id_arr) and (sample[counter-1] != vocabulary_size-3):
				sample[counter] = vocabulary_size-3
				counter += 1
			elif (words[1] in tx_id_arr) and (sample[counter-1] != vocabulary_size-2):
				sample[counter] = vocabulary_size-2
				counter += 1
			elif words[0] in dictionary.keys() and dictionary[words[0]] != sample[counter-1]:
				sample[counter] = dictionary[words[0]]
				counter += 1
			elif words[0] not in dictionary.keys():
				sample[counter] = dictionary['UNK']
				counter += 1
			if counter >= max_words-1:
				break
				# while going through, see if conceptid associated then add to dataframe with root index and rel_type index
		if is_test:
			final_results = final_results.append(pd.DataFrame([[sentence['sentence'],sentence['sentence_tuples'], item['condition_id'], item['treatment_id'], sample, item['label']]], columns=['sentence', 'sentence_tuples', 'condition_id', 'treatment_id', 'x_train', 'label']))
		else:
			final_results = final_results.append(pd.DataFrame([[sentence['sentence'],sentence['sentence_tuples'], item['condition_id'], item['treatment_id'], sample]], columns=['sentence', 'sentence_tuples', 'condition_id', 'treatment_id', 'x_train']))

	return final_results

def build_model():
	# pd.set_option('display.max_colwidth', -1)
	training_set = pd.read_pickle("./training__02_10_19")
	training_set = training_set.sample(frac=1).reset_index(drop=True)
	test_set = pd.read_pickle("./testing__02_10_19")

	x_train = np.array(training_set['x_train'].tolist())
	y_train = np.array(training_set['label'].tolist())

	x_test = np.array(test_set['x_train'].tolist())
	y_test = np.array(test_set['label'].tolist())
	
	embedding_size=32
	batch_size = 64
	num_epochs = 3
	model=Sequential()
	model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
	model.add(Conv1D(filters=32, kernel_size=10, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(LSTM(100))
	# model.add(LSTM(100, return_sequences=True, input_shape=(embedding_size, batch_size)))
	# model.add(LSTM(100, return_sequences=True))
	# model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))

	print(model.summary())
	model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])
	
	x_valid, y_valid = x_train[:batch_size], y_train[:batch_size]
	x_train2, y_train2 = x_train[batch_size:], y_train[batch_size:]

	print(x_train2.shape)
	print(y_train2.shape)
	
	model.fit(x_train2, y_train2, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

	print(x_test)
	scores = model.evaluate(x_test, y_test, verbose=1)
	model.save('txp_60_02_011.2.h5')
	print(scores)
	print('Test accuracy:', scores[1])
	
	
	model = load_model('txp_60_02_011.2.h5')

	correct_counter = 0
	counter = 0
	for i,d in test_set.iterrows():
		
		prediction = model.predict(np.array([d['x_train']]))
		

		if ((prediction > 0.50) and (d['label'] == 0)) or ((prediction <= 0.50) and (d['label'] == 1)):
			print(d['label'])
			print(prediction)
			print(d['sentence'])
			print(d['x_train'])
			print("condition_id : " + str(d['condition_id']) + " treatment_id : " + str(d['treatment_id']))

			counter+=1
			print("=========" + str(i))
		else:
			correct_counter += 1
		if counter >= 100:
			break
		
		
	print(correct_counter)
	print(len(test_set[test_set['label'] == 1]))
	print(Counter(test_set['label'].tolist()))
	# for i,d in enumerate(x_test):
	# 	print(model.predict(np.array([d])))
	# 	print(y_test[i])
	# 	print(raw_test.iloc[i].to_string())
	# 	print(test_sentences[i])
	# 	print("=========" + str(i))
	# 	if i == 100:
	# 		break
	# cd = np.array([x_test[1301]])
	# print(model.predict(cd))
	# print(y_train[1301])
	# scores = model.evaluate(steps=1)
			
			# word_list.append(words[0])
			# if words[1] in root then number is vocabulary_size-2
			# if words[1] in rel then number is vocabulary_size-1
			# if words[0] in dictionary, then number
			# else then position for UNK.
# 1) pull sentences with both concept types
# 2) figure out how to get word sequences within range that still will include both conceptids

def confirmation():
	conn,cursor = pg.return_postgres_cursor()
	labelled_query = "select condition_id, treatment_id, label from annotation.labelled_treatments_confirmation"
	labelled_ids = pg.return_df_from_query(cursor, labelled_query, None, ["condition_id", "treatment_id", "label"])
	labelled_ids.to_pickle("./confirmation_ids")

	get_labelled_data_from_files("./confirmation_ids", "./confirmation_60.pkl")
	test_set = pd.read_pickle("./confirmation_60.pkl")

	x_test = np.array(test_set['x_train'].tolist())
	y_test = np.array(test_set['label'].tolist())
	model = load_model('txp_60_02_11_glove.2.h5')

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
	conditions_query = """select conceptid, tb1.count
						from annotation.concept_counts tb1
						left join (select * from 
								(select root_cid, rel_type, active, row_number () over (partition by root_cid order by effectivetime desc) as row_num 
		   						from annotation.concept_types) tb1 where row_num=1) tb2
						on tb1.conceptid=tb2.root_cid
						where tb2.rel_type = 'condition'
						order by tb1.count desc limit 10"""
	conditions_df = pg.return_df_from_query(cursor, conditions_query, None, ["conceptid", "count"])
	conditions_df.columns = ['condition_cid', 'count']

	raw_results = pd.DataFrame()
	for i,c in conditions_df.iterrows():

		treatments_query = """
							select distinct(conceptid)
							from annotation.sentences3
							where id in (select id from annotation.sentences3 where conceptid=%s)
							and section in ('objective', 'title', 'conclusions', 'background', 'unlabelled', 'unassigned')
					"""
		treatments_df = pg.return_df_from_query(cursor, treatments_query, (c['condition_cid'],), ['conceptid'])
		treatments_df.columns = ['treatment_cid']
		print("treatments df length : " + str(len(treatments_df)))

		for i2,t in treatments_df.iterrows():
			print(i2)
			results_df = pd.DataFrame()
			sentences_df = get_sentences_df([c['condition_cid']], [t['treatment_cid']], cursor)
			print("len sentences df : " + str(len(sentences_df)))
			results_df = results_df.append(get_labelled_data(False, sentences_df, [c['condition_cid']], [t['treatment_cid']], pd.DataFrame([[c['condition_cid'], t['treatment_cid']]], columns=["condition_id", "treatment_id"]), dictionary, reverse_dictionary), sort=False, ignore_index=True)
			results_df.reset_index()

			results_df['score'] = 0.0

			for i3,s in results_df.iterrows():
				score = model.predict(np.array([s['x_train']]))[0][0]
				results_df.loc[i3, 'score'] = score
				raw_results = raw_results.append(pd.DataFrame([[s['condition_id'].values[0], s['treatment_id'].values[0], s['sentence'], score]], columns=['condition_id', 'treatment_id', 'sentence', 'score']))
			
	
	u.pprint(raw_results)
	engine = pg.return_sql_alchemy_engine()
	raw_results.to_sql('raw_treatment_recs', engine, schema='annotation', if_exists='replace', index=False)
	cursor.close()

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
	training_set = pd.read_pickle("./training__02_10_19")
	training_set = training_set.sample(frac=1).reset_index(drop=True)
	test_set = pd.read_pickle("./testing__02_10_19")

	x_train = np.array(training_set['x_train'].tolist())
	y_train = np.array(training_set['label'].tolist())

	x_test = np.array(test_set['x_train'].tolist())
	y_test = np.array(test_set['label'].tolist())
	
	embedding_size=100
	batch_size = 64
	num_epochs = 3
	model=Sequential()
	model.add(Embedding(vocabulary_size, embedding_size, weights=[embedding_matrix], input_length=max_words))
	model.add(Conv1D(filters=32, kernel_size=10, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	# model.add(LSTM(100))
	model.add(LSTM(100, return_sequences=True, input_shape=(embedding_size, batch_size)))
	model.add(LSTM(100, return_sequences=True))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))

	print(model.summary())
	model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])
	
	x_valid, y_valid = x_train[:batch_size], y_train[:batch_size]
	x_train2, y_train2 = x_train[batch_size:], y_train[batch_size:]

	print(x_train2.shape)
	print(y_train2.shape)
	
	model.fit(x_train2, y_train2, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

	scores = model.evaluate(x_test, y_test, verbose=1)
	model.save('txp_60_02_11_glove.2.h5')
	print(scores)
	print('Test accuracy:', scores[1])
	
	
	model = load_model('txp_60_02_11_glove.2.h5')

	correct_counter = 0
	counter = 0
	for i,d in test_set.iterrows():
		
		prediction = model.predict(np.array([d['x_train']]))
		

		if ((prediction > 0.50) and (d['label'] == 0)) or ((prediction <= 0.50) and (d['label'] == 1)):
			print(d['label'])
			print(prediction)
			print(d['sentence'])
			print(d['x_train'])
			print("condition_id : " + str(d['condition_id']) + " treatment_id : " + str(d['treatment_id']))

			counter+=1
			print("=========" + str(i))
		else:
			correct_counter += 1
		if counter >= 100:
			break
		
	print(correct_counter)
	print(len(test_set[test_set['label'] == 1]))
	print(Counter(test_set['label'].tolist()))

# glove_embeddings()

confirmation()
# treatment_recategorization_recs('txp_60_02_11_glove.2.h5')

# gen_datasets()
# gen_datasets_2("_02_10_19")
# get_labelled_data_from_files("./testing_ids_02_08_19", "./test_set_60_02_08_19.pkl")
# get_labelled_data_from_files("./training_ids_02_08_19", "./training_set_60_02_08_19.pkl")
# build_model()
