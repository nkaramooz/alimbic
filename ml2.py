import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from nltk.stem.wordnet import WordNetLemmatizer
from operator import itemgetter
import nltk.data
import numpy as np
import time
import utilities.utils as u, utilities.pglib as pg
import collections
import os
import random
import sys
import snomed_annotator2 as ann2
import pickle as pk
import sys
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten
from keras.models import load_model
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import snomed_annotator as ann
from numpy import array 
from keras.layers import GlobalMaxPooling1D
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
import sqlalchemy as sqla
import multiprocessing as mp


# index 0 = filler
# index 1 = UNK

spacer_ind = 0
UNK_ind = 1
vocabulary_size = 50000


# Vocabulary spacer should be 4
vocabulary_spacer = 4
target_treatment_key = vocabulary_size-vocabulary_spacer+3 #49,999
target_condition_key = vocabulary_size-vocabulary_spacer+2 #49,998
generic_condition_key = vocabulary_size-vocabulary_spacer+1 #49,997
generic_treatment_key = vocabulary_size-vocabulary_spacer #49,996

max_words = 60

# bounded
b_target_treatment_key_start = target_treatment_key # 49997
b_target_treatment_key_end = vocabulary_size+1 # 50001
b_generic_treatment_key_start = generic_treatment_key # 4995
b_generic_treatment_key_stop = vocabulary_size+2 # 50002


def get_all_conditions_set():
	query = "select root_acid from annotation2.base_concept_types where rel_type='condition'"
	conn,cursor = pg.return_postgres_cursor()
	all_conditions_set = set(pg.return_df_from_query(cursor, query, None, ['root_acid'])['root_acid'].tolist())
	return all_conditions_set

def get_all_treatments_set():
	
	query = "select root_acid from annotation2.base_concept_types where rel_type='treatment'"
	conn,cursor = pg.return_postgres_cursor()
	all_treatments_set = set(pg.return_df_from_query(cursor, query, None, ['root_cid'])['root_cid'].tolist())
	return all_treatments_set


def get_word_index(word, cursor):
	query = "select rn from ml2.word_counts_50k where word=%s limit 1"
	ind_df = pg.return_df_from_query(cursor, query, (word,), ['rn'])
	
	# return index for UNK
	if (len(ind_df.index) == 0):
		return UNK_ind
	else:
		return int(ind_df['rn'][0])

def get_word_from_ind(index):
	if index == UNK_ind:
		return 'UNK'
	else:
		conn,cursor = pg.return_postgres_cursor()
		query = "select word from ml2.word_counts_50k where rn = %s limit 1"
		word_df = pg.return_df_from_query(cursor, query, (index,), ['word'])
		return str(word_df['word'][0])


def train_data_generator(batch_size, cursor):
	print("train_data_generator")
	while True:
		curr_version = int(pg.return_df_from_query(cursor, "select min(ver_gen) from ml2.train_sentences", \
			None, ['ver_gen'])['ver_gen'][0])

		# Note that ordering by random adds significant time
		new_version = curr_version + 1
		query = "select id, x_train_gen, label from ml2.train_sentences where ver_gen = %s limit %s"
		train_df = pg.return_df_from_query(cursor, query, (curr_version, batch_size), ['id', 'x_train_gen', 'label'])

		x_train_gen = train_df['x_train_gen'].tolist()
		y_train = train_df['label'].tolist()
		id_df = train_df['id'].tolist()

		try:
			query = """
				UPDATE ml2.train_sentences
				SET ver_gen = %s
				where id = ANY(%s);
			"""
			cursor.execute(query, (new_version, id_df))
			cursor.connection.commit()

			yield (np.asarray(x_train_gen), np.asarray(y_train))
		except:
			print("update version failed. Rolling back")
			cursor.connection.rollback()
			yield None


# Below not working, likely due to need to concatenate
def train_data_generator_v2(batch_size, cursor):

	while True:
		curr_version = int(pg.return_df_from_query(cursor, "select min(ver_gen) from ml2.train_sentences", \
			None, ['ver_gen'])['ver_gen'][0])

		# Note that ordering by random adds significant time
		new_version = curr_version + 1
		query = "select id, section_ind, x_train_gen, label from ml2.train_sentences where ver_gen = %s limit %s"
		train_df = pg.return_df_from_query(cursor, query, (curr_version, batch_size), ['id', 'section_ind', 'x_train_gen', 'label'])

		x_train_gen = train_df['x_train_gen'].tolist()
		section_ind = [0]* max_words
		section_ind[0] = train_df['section_ind'].values[0]
		y_train = train_df['label'].tolist()
		id_df = train_df['id'].tolist()

		try:
			query = """
				UPDATE ml2.train_sentences
				SET ver_gen = %s
				where id = ANY(%s);
			"""
			cursor.execute(query, (new_version, id_df))
			cursor.connection.commit()
			# print([np.asarray(x_train_gen), np.asarray(section_ind)], np.asarray(y_train))
			yield [np.asarray(x_train_gen), np.asarray(section_ind)], np.asarray(y_train)
			# print((np.asarray(x_train_gen), np.asarray(y_train)))
			# yield (np.asarray(x_train_gen), np.asarray(y_train))
		except:
			print("update version failed. Rolling back")
			cursor.connection.rollback()
			yield None

	
def train_with_word2vec():
	conn,cursor = pg.return_postgres_cursor()
	report = open('ml_report.txt', 'w')
	embedding_size=500
	batch_size = 500
	num_epochs = 10

	model=Sequential()
	model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words, trainable=True))
	model.add(LSTM(500, return_sequences=True, input_shape=(embedding_size, batch_size)))
	model.add(Dropout(0.3))
	model.add(TimeDistributed(Dense(500)))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))

	print(model.summary())
	model.summary(print_fn=lambda x: report.write(x + '\n'))
	report.write('\n')

	model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])
	checkpointer = ModelCheckpoint(filepath='./model-{epoch:02d}.hdf5', verbose=1)

	history = model.fit_generator(train_data_generator(batch_size, cursor),
	 epochs=num_epochs, class_weight={0:1, 1:50}, steps_per_epoch =((4993115//batch_size)+1),
	 callbacks=[checkpointer])

	report.close()

def train_with_word2vec_v2():
	conn,cursor = pg.return_postgres_cursor()
	# report = open('ml_report.txt', 'w')
	embedding_size=500
	batch_size = 500
	num_epochs = 100

	# model=Sequential()
	# model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words, trainable=True))
	# model.add(Bidirectional(LSTM(500, dropout=0.3, return_sequences=True, input_shape=(embedding_size, batch_size))))
	# model.add(TimeDistributed(Dense(50)))
	# model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
	# model.add(Dense(20))
	# model.add(Flatten())
	# model.add(Dense(1, activation='sigmoid'))

	inp1 = Input(shape=(max_words,))
	first_model = Embedding(vocabulary_size, embedding_size, input_length=max_words, trainable=True)(inp1)
	first_model = Bidirectional(LSTM(500, dropout=0.3, recurrent_dropout=0.3, return_sequences=True, input_shape=(embedding_size, batch_size)))(first_model)
	first_model = TimeDistributed(Dense(100))(first_model)
	first_model = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(first_model)
	first_model = Dropout(0.3)(first_model)
	first_model = Dense(25, activation='relu')(first_model)
	first_model = Model(inputs=[inp1], outputs=first_model)



	# first_model = Dropout(0.3)(first_model)
	# first_model = Dense(10, activation='relu')(first_model)
	# first_model = Flatten()(first_model)
	# first_model = Dense(1, activation='sigmoid')(first_model)



	# inp1 = Input(shape=(max_words,))
	# first_model = Embedding(vocabulary_size, embedding_size, input_length=max_words, trainable=True)(inp1)
	# first_model = LSTM(50, return_sequences=True, input_shape=(embedding_size, batch_size))(first_model)

	# first_model = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(first_model)
	# first_model = Dropout(0.3)(first_model)
	# first_model = Dense(25, activation='relu')(first_model)



	inp2 = Input(shape=(max_words,))
	second_model = Embedding(vocabulary_size, embedding_size, input_length=max_words, trainable=True)(inp2)
	# second_model = LSTM(50, return_sequences=True, input_shape=(embedding_size, batch_size))(second_model)

	# second_model = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inp2)
	# second_model = Dropout(0.3)(second_model)
	second_model = Dense(25, activation='relu')(second_model)
	second_model = Model(inputs=[inp2], outputs=second_model)


	c = Concatenate()(inputs=[first_model.output, second_model.output])
	out = Embedding(vocabulary_size, embedding_size, trainable=True)(c)
	out = Bidirectional(LSTM(100))(c)
	out = Dense(50, activation='relu')(c)
	out = Dropout(0.3)(out)
	out = Dense(10, activation='relu')(out)
	out = Flatten()(out)
	out = Dense(1, activation='sigmoid')(out)
	model = Model(inputs=[inp1, inp2], outputs=out)

	print(model.summary())
	# model.summary(print_fn=lambda x: report.write(x + '\n'))
	# report.write('\n')

	model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

	checkpointer = ModelCheckpoint(filepath='./double-{epoch:02d}.hdf5', verbose=1)

	
	history = model.fit(train_data_generator_v2(batch_size, cursor), \
	 epochs=num_epochs, class_weight={0:1, 1:50}, steps_per_epoch =((4142230//batch_size)+1),
	  callbacks=[checkpointer])


def update_word2vec(model_name):
	conn,cursor = pg.return_postgres_cursor()
	embedding_size=500
	batch_size = 500
	num_epochs = 4
	model = load_model(model_name)

	all_conditions_set = get_all_conditions_set()
	all_treatments_set = get_all_treatments_set()

	
	checkpointer = ModelCheckpoint(filepath='./model010419-{epoch:02d}.hdf5', verbose=1)


	history = model.fit_generator(train_data_generator(batch_size, cursor), \
	 epochs=num_epochs, class_weight={0:1, 1:5}, steps_per_epoch =((2884571//batch_size)+1), callbacks=[checkpointer])

	

def parallel_treatment_recategorization_top(model_name):
	conn,cursor = pg.return_postgres_cursor()
	
	all_conditions_set = get_all_conditions_set() 
	all_treatments_set = get_all_treatments_set()


	query = "select min(ver) from ml2.treatment_candidates"
	new_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])+1
	old_version = new_version-1
	curr_version = old_version

	while curr_version != new_version:
		parallel_treatment_recategorization_bottom(model_name, old_version, all_conditions_set, all_treatments_set)
		query = "select min(ver) from ml2.treatment_candidates"
		curr_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])


def parallel_treatment_recategorization_bottom(model_name, old_version, all_conditions_set, all_treatments_set):
	conn,cursor = pg.return_postgres_cursor()
	number_of_processes = 30

	task_queue = mp.Queue()

	pool = []
	for i in range(number_of_processes):
		p = mp.Process(target=recat_worker, args=(task_queue,))
		pool.append(p)
		p.start()

	counter = 0

	query = """
		select 
			sentence_id
			,condition_acid
			,treatment_acid
			,sentence_tuples
		from ml2.treatment_candidates
		where ver = %s limit 5000"""

	treatment_candidates_df = pg.return_df_from_query(cursor, query, \
				(old_version,), ['sentence_id', 'condition_acid', 'treatment_acid', 'sentence_tuples'])

	while (counter < number_of_processes) and (len(treatment_candidates_df.index) > 0):
		params = (model_name, treatment_candidates_df, all_conditions_set, all_treatments_set)
		task_queue.put((batch_treatment_recategorization, params))

		update_query = """
			set schema 'ml2';
			UPDATE treatment_candidates
			SET ver = %s
			where sentence_id = ANY(%s) and condition_acid = ANY(%s) and treatment_acid = ANY(%s);
		"""

		sentence_id_list = treatment_candidates_df['sentence_id'].tolist()
		condition_acid_list = treatment_candidates_df['condition_acid'].tolist()
		treatment_acid_list = treatment_candidates_df['treatment_acid'].tolist()
		new_version = old_version + 1
		cursor.execute(update_query, (new_version, sentence_id_list, condition_acid_list, treatment_acid_list))
		cursor.connection.commit()
		treatment_candidates_df = pg.return_df_from_query(cursor, query, \
			(old_version,), ['sentence_id', 'condition_acid', 'treatment_acid', 'sentence_tuples'])

		counter += 1

	for i in range(number_of_processes):
		task_queue.put('STOP')

	for p in pool:
		p.join()


def recat_worker(input):
	for func,args in iter(input.get, 'STOP'):
		recat_calculate(func, args)


def recat_calculate(func, args):
	func(*args)


def apply_get_generic_labelled_data(row, all_conditions_set, all_treatments_set):
	return get_labelled_data_sentence_generic_v2(row, all_conditions_set, all_treatments_set)

def apply_get_specific_labelled_data(row):
	return get_labelled_data_sentence_specific_v2(row, row['condition_acid'], row['treatment_acid'])

def apply_score(row, model):
	# print([row['x_train_gen'], row['x_train_spec']])
	gen = np.array([row['x_train_gen']])
	# spec = np.array([row['x_train_spec']])

	res = float(model.predict([gen])[0][0])
	return res


def batch_treatment_recategorization(model_name, treatment_candidates_df, all_conditions_set, all_treatments_set):
	model = load_model(model_name)
	conn,cursor = pg.return_postgres_cursor()
	engine = pg.return_sql_alchemy_engine()

	treatment_candidates_df['x_train_gen'] = treatment_candidates_df.apply(apply_get_generic_labelled_data, \
		all_conditions_set=all_conditions_set, all_treatments_set=all_treatments_set, axis=1)
	
	treatment_candidates_df['score'] = treatment_candidates_df.apply(apply_score, model=model, axis=1)
	treatment_candidates_df.to_sql('treatment_recs_staging', engine, schema='ml2', if_exists='append', index=False, dtype={'sentence_tuples' : sqla.types.JSON})

	cursor.close()
	conn.close()
	engine.dispose()



# sample[0] = dict value of item
# sample[1]: 1=target_condition, 2=target_treatment
def get_labelled_data_sentence_generic_v2(sentence, conditions_set, all_treatments_set):
	conn,cursor = pg.return_postgres_cursor()
	condition_id = sentence['condition_acid']
	tx_id = sentence['treatment_acid']

	final_results = pd.DataFrame()
	sample = [0]*max_words
	counter = 0

	for index,words in enumerate(sentence['sentence_tuples']):		
		if words[1] == condition_id and sample[counter-1] == target_condition_key:
			continue
		elif words[1] == condition_id and sample[counter-1] != target_condition_key:
			sample[counter] = target_condition_key
		elif (words[1] in conditions_set) and (sample[counter-1] == generic_condition_key):
			continue
		elif (words[1] in conditions_set) and (sample[counter-1] != generic_condition_key):
			sample[counter] = generic_condition_key
		elif (words[1] == tx_id and (sample[counter-1] == target_treatment_key)):
			continue
		elif (words[1] == tx_id) and (sample[counter-1] != target_treatment_key):
			sample[counter] = target_treatment_key
		elif (words[1] in all_treatments_set) and (sample[counter-1] == generic_treatment_key):
			continue
		elif (words[1] in all_treatments_set) and (sample[counter-1] != generic_treatment_key):
			sample[counter] = generic_treatment_key
		elif (words[1] != 0) and (get_word_index(words[1], cursor) != UNK_ind) and (sample[counter-1] != get_word_index(words[1], cursor)):
			sample[counter] = get_word_index(words[1], cursor)
		elif (words[1] == 0) and (get_word_index(words[0], cursor) != UNK_ind) and (sample[counter-1] != get_word_index(words[0], cursor)):
			sample[counter] = get_word_index(words[0], cursor)
		elif ((words[1] != 0 and (get_word_index(words[1], cursor) == UNK_ind)) or (words[1] == 0 and get_word_index(words[0], cursor) == UNK_ind)):
			sample[counter] = UNK_ind
		else:
			counter -= 1

		counter += 1

		if counter >= max_words-1:
			break

	cursor.close()
	conn.close()
	return sample

def get_labelled_data_sentence_custom_v2(sentence, condition_id, tx_id):
	conn,cursor = pg.return_postgres_cursor()
	final_results = pd.DataFrame()
	
	# outer_counter will keep track of length of tx string
	outer_counter = 1
	#tx_counter will keep track of where you are within tx_string
	tx_counter_target = 0
	tx_counter = 0
	while (tx_counter < outer_counter):
	## may need to add 0
		counter = 0
		tx_counter_current = 0
		sample = [0]*max_words
		for index,words in enumerate(sentence['sentence_tuples']):

			if words[1] == condition_id and sample[counter-1] == target_condition_key:
				continue
			elif words[1] == condition_id and sample[counter-1] != target_condition_key:
				sample[counter] = target_condition_key
			elif (words[1] == tx_id):
				if tx_counter_current == tx_counter_target:
					sample[counter] = target_treatment_key
				elif (sample[counter-1] != get_word_index(words[1], cursor)):
					sample[counter] = get_word_index(words[1], cursor)
				else:
					counter -= 1

				if tx_counter_target == 0 and tx_counter_current != 0:
					outer_counter += 1
				tx_counter_current += 1
			elif (words[1] != 0) and (get_word_index(words[1], cursor) != UNK_ind) and (sample[counter-1] != get_word_index(words[1], cursor)):
				sample[counter] = get_word_index(words[1], cursor)
			elif (words[1] == 0) and (get_word_index(words[0], cursor) != UNK_ind) and (sample[counter-1] != get_word_index(words[0], cursor)):
				sample[counter] = get_word_index(words[0], cursor)
			elif ((words[1] == 0) and (get_word_index(words[0], cursor) == UNK_ind)) or ((words[1] == 1) \
				and (get_word_index(words[1], cursor) == UNK_ind)):
				sample[counter] = UNK_ind
			elif ((words[1] != 0 and (get_word_index(words[1], cursor) == UNK_ind)) or (words[1] == 0 and get_word_index(words[0], cursor) == UNK_ind)):
				sample[counter] = UNK_ind
			else:
				counter -= 1

			counter += 1

			if counter >= max_words-1:
				break
		tx_counter_target += 1
		final_results = final_results.append(pd.DataFrame([[sentence['id'], sentence['sentence'], sentence['sentence_tuples'],\
			condition_id, tx_id, sample]]))
		tx_counter += 1
	
	cursor.close()
	conn.close()
	final_results.columns = ['id', 'sentence', 'sentence_tuples', 'condition_id', 'treatment_id', 'x_train_cust']

	return final_results


def get_labelled_data_sentence_specific_v2(sentence, condition_id, tx_id):
	conn,cursor = pg.return_postgres_cursor()
	final_results = pd.DataFrame()
	sample = [0]*max_words 
	counter = 0
	
	for index,words in enumerate(sentence['sentence_tuples']):
		if words[1] == condition_id and sample[counter-1] == get_word_index(words[1], cursor):
			continue
		elif words[1] == condition_id and sample[counter-1] != get_word_index(words[1], cursor):
			sample[counter] = get_word_index(words[1], cursor)
		elif (words[1] == tx_id) and (sample[counter-1] == get_word_index(words[1], cursor)):
			continue
		elif (words[1] == tx_id) and (sample[counter-1] != get_word_index(words[1], cursor)):
			sample[counter] = get_word_index(words[1], cursor)
		elif (words[1] != 0) and (get_word_index(words[1], cursor) != UNK_ind) and (sample[counter-1] != get_word_index(words[1], cursor)):
			sample[counter] = get_word_index(words[1], cursor)
		elif (words[1] == 0) and (get_word_index(words[0], cursor) != UNK_ind) and (sample[counter-1] != get_word_index(words[0], cursor)):
			sample[counter] = get_word_index(words[0], cursor)
		elif ((words[1] != 0 and (get_word_index(words[1], cursor) == UNK_ind)) or (words[1] == 0 and get_word_index(words[0], cursor) == UNK_ind)):
			sample[counter] = UNK_ind
		else:
			counter -= 1

		counter += 1

		if counter >= max_words-1:
			break

	cursor.close()
	conn.close()

	return sample


def get_labelled_data_sentence_specific_v2_custom(sentence, condition_id, tx_word, conditions_set, get_all_treatments_set):
	conn,cursor = pg.return_postgres_cursor()
	final_results = pd.DataFrame()
	sample = [0]*max_words

	counter = 0

	for index,words in enumerate(sentence['sentence_tuples'][0]):
		if words[1] == condition_id and sample[counter-1] == get_word_index(words[1], cursor):
			continue
		elif words[1] == condition_id and sample[counter-1] != get_word_index(words[1], cursor):
			sample[counter] = get_word_index(words[1], cursor)

		elif (words[0] == tx_word):
			sample[counter] = get_word_index(words[0], cursor)

		elif (words[1] != 0) and (get_word_index(words[1], cursor) != UNK_ind) and (sample[counter-1] != get_word_index(words[1], cursor)):
			sample[counter] = get_word_index(words[1], cursor)
		elif (words[1] == 0) and (get_word_index(words[0], cursor) != UNK_ind) and (sample[counter-1] != get_word_index(words[0], cursor)):
			sample[counter] = get_word_index(words[0], cursor)
		elif ((words[1] == 0) and (get_word_index(words[0], cursor) == UNK_ind)) or ((words[1] == 1) \
			and (get_word_index(words[1], cursor) == UNK_ind)):
			sample[counter] = UNK_ind
		else:
			counter -= 1

		counter += 1

		if counter >= max_words-1:
			break

	cursor.close()
	conn.close()

	return sample

def get_bounded_labelled_data_sentence(sentence, condition_id, tx_id, conditions_set, all_treatments_set):
	conn,cursor = pg.return_postgres_cursor()
	final_results = pd.DataFrame()
	sample = [0]*max_words
	counter=0
	for index,words in enumerate(sentence['sentence_tuples']):
		## words[1] is the conceptid, words[0] is the word
		if words[1] == condition_id and sample[counter-1] == target_condition_key:
			continue
		elif words[1] == condition_id and sample[counter-1] != target_condition_key:
			sample[counter] = target_condition_key
		elif (words[1] in conditions_set) and (sample[counter-1] != generic_condition_key):
			sample[counter] = generic_condition_key
		elif (words[1] == tx_id) and (sample[counter-1] != target_treatment_key):
			sample[counter] = b_target_treatment_key_start
			counter += 1
			if counter >= max_words-1:
				break
			sample[counter] = get_word_index(words[1], cursor)
			counter += 1
			if counter >= max_words-1:
				break
			sample[counter] = b_target_treatment_key_end
		elif (words[1] in all_treatments_set) and (sample[counter-1] != b_generic_treatment_key_stop):
			sample[counter] = b_generic_treatment_key_start
			counter += 1
			if counter >= max_words-1:
				break
			sample[counter] = get_word_index(words[1], cursor)
			counter += 1
			if counter >= max_words-1:
				break
			sample[counter] = b_generic_treatment_key_stop
			# Now using conceptid if available
		elif words[1] != 0 and (get_word_index(words[1], cursor) != UNK_ind) and get_word_index(words[1], cursor) != sample[counter-1]:
			sample[counter] = get_word_index(words[1], cursor)
		elif (words[1] == 0 and (get_word_index(words[0], cursor) != UNK_ind) and get_word_index(words[0], cursor) != sample[counter-1])\
			and words[1] not in conditions_set and words[1] not in all_treatments_set:
			sample[counter] = get_word_index(words[0], cursor)
		elif (words[1] == 0) and (get_word_index(words[0], cursor) == UNK_ind) and words[1] not in conditions_set and words[1] not in all_treatments_set:
			sample[counter] = get_word_index(words[0], cursor)
		else:
			counter -= 1
		
		counter += 1

		if counter >= max_words-1:
			break
	cursor.close()
	conn.close()
	return sample

def gen_datasets_top(write_type):
	query = "select min(ver) as ver from ml2.labelled_treatments t1 where label=0 or label=1"
	new_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])+1
	old_version = new_version-1
	conditions_set = get_all_conditions_set()
	treatments_set = get_all_treatments_set()


	query = "select count(*) as cnt from ml2.labelled_treatments where label=0 or label=1"
	max_counter = int(pg.return_df_from_query(cursor, query, None, ['cnt'])['cnt'][0])

	query = """
			select t1.id, condition_acid, treatment_acid, label, ver from ml2.labelled_treatments t1 where ver=%s and (label=0 or label=1) limit 1
		"""
	labels_df = pg.return_df_from_query(cursor, query, (old_version,), ['id', 'condition_acid', 'treatment_acid', 'label', 'ver'])

	
	while (len(labels_df) > 0):
		write_sentence_vectors_from_labels(labels_df, conditions_set, treatments_set, write_type)
		query = "UPDATE ml2.labelled_treatments set ver = %s where id = %s"
		cursor.execute(query, (new_version, labels_df['id'].values[0]))
		cursor.connection.commit()

		query = """
			select t1.id, condition_acid, treatment_acid, label, ver from ml2.labelled_treatments t1 where ver=%s and (label=0 or label=1) limit 1
		"""
		labels_df = pg.return_df_from_query(cursor, query, (old_version,), ['id', 'condition_acid', 'treatment_acid', 'label', 'ver'])


def gen_datasets_mp_bottom(condition_acid, treatment_acid, label, conditions_set, treatments_set):
	conn,cursor = pg.return_postgres_cursor()
	sentences_df = pd.DataFrame()

	if condition_acid == '%':
		tx_id_arr = [treatment_acid]
		tx_id_arr.extend(ann2.get_children(treatment_acid, cursor))
		
		sentences_query = """
						select
							t4.sentence_tuples
							,t2.condition_acid
							,t3.treatment_acid
							,t1.sentence_id
							,t1.section_ind
							,t1.pmid
						from pubmed.sentence_concept_arr t1
						join (select root_acid as condition_acid from annotation2.base_concept_types where rel_type='condition' or rel_type='symptom') t2
							on t2.condition_acid = ANY(t1.concept_arr::text[])
						join (select acid as treatment_acid from annotation2.downstream_root_cid where acid in %s) t3
							on t3.treatment_acid = ANY(t1.concept_arr::text[])
						join pubmed.sentence_tuples t4
							on t1.sentence_id = t4.sentence_id
					"""
		sentences_df = pg.return_df_from_query(cursor, sentences_query, (tuple(tx_id_arr),), \
				['sentence_tuples', 'condition_acid', 'treatment_acid', 'sentence_id', 'section_ind', 'pmid'])
			
		print("finished query")
			
		print("len of wildcard: " + str(len(sentences_df)))
	else:
		condition_id_arr = [condition_acid]
		condition_id_arr.extend(ann2.get_children(condition_acid, cursor))

		tx_id_arr = [treatment_acid]
		tx_id_arr.extend(ann2.get_children(treatment_acid, cursor))
		sentences_query = """
				select
					t4.sentence_tuples
					,t2.condition_acid
					,t3.treatment_acid
					,t1.sentence_id
					,t1.section_ind
					,t1.pmid
				from pubmed.sentence_concept_arr t1
				join (select acid as condition_acid from annotation2.downstream_root_cid where acid in %s) t2
					on t2.condition_acid = ANY(t1.concept_arr::text[])
				join (select acid as treatment_acid from annotation2.downstream_root_cid where acid in %s) t3
					on t3.treatment_acid = ANY(t1.concept_arr::text[])
				join pubmed.sentence_tuples t4
					on t1.sentence_id = t4.sentence_id
			"""
		sentences_df = pg.return_df_from_query(cursor, sentences_query, (tuple(condition_id_arr), tuple(tx_id_arr)), \
				['sentence_tuples', 'condition_acid', 'treatment_acid', 'sentence_id', 'section_ind', 'pmid'])
	cursor.close()
	conn.close()

	if len(sentences_df.index) > 0:
		sentences_df['label'] = label
		write_sentence_vectors_from_labels(sentences_df, conditions_set, treatments_set, 'gen')


def gen_datasets_mp(new_version):
	number_of_processes = 48
	old_version = new_version-1
	conditions_set = get_all_conditions_set()
	treatments_set = get_all_treatments_set()

	task_queue = mp.Queue()
	pool = []

	for i in range(number_of_processes):
		p = mp.Process(target=gen_dataset_worker, args=(task_queue, conditions_set, treatments_set))
		pool.append(p)
		p.start()

	conn,cursor = pg.return_postgres_cursor()

	query = """
			select t1.id, condition_acid::text, treatment_acid::text, label, ver from ml2.labelled_treatments t1 where ver=%s and (label=0 or label=1) limit %s
		"""
	labels_df = pg.return_df_from_query(cursor, query, (old_version, number_of_processes), ['id', 'condition_acid', 'treatment_acid', 'label', 'ver'])
	

	while len(labels_df.index) > 0:
		for index,item in labels_df.iterrows():
			params = (item['condition_acid'], item['treatment_acid'], item['label'])
			task_queue.put((gen_datasets_mp_bottom, params))

		query = "UPDATE ml2.labelled_treatments set ver = %s where id in %s"
		cursor.execute(query, (new_version, tuple(labels_df['id'].values.tolist())))
		cursor.connection.commit()

		query = """
			select t1.id, condition_acid, treatment_acid, label, ver from ml2.labelled_treatments t1 where ver=%s and (label=0 or label=1) limit %s
		"""
		labels_df = pg.return_df_from_query(cursor, query, (old_version, number_of_processes), ['id', 'condition_acid', 'treatment_acid', 'label', 'ver'])
	
		
	for i in range(number_of_processes):
		task_queue.put('STOP')

	for p in pool:
		p.join()

	for p in pool:
		p.close()

	cursor.close()
	conn.close()


def gen_dataset_worker(input, conditions_set, treatments_set):
	for func,args in iter(input.get, 'STOP'):
		gen_dataset_calculate(func, args, conditions_set, treatments_set)

def gen_dataset_calculate(func, args, conditions_set, treatments_set):
	func(*args, conditions_set, treatments_set)


def write_sentence_vectors_from_labels(sentences_df, conditions_set, treatments_set, write_type):
	conn,cursor = pg.return_postgres_cursor()
	engine = pg.return_sql_alchemy_engine()

	if len(sentences_df.index) > 0:
		if write_type == 'gen':
			sentences_df['x_train_gen'] = sentences_df.apply(apply_get_generic_labelled_data, all_conditions_set=conditions_set, all_treatments_set=treatments_set, axis=1)
			sentences_df['x_train_spec'] = sentences_df.apply(apply_get_specific_labelled_data, axis=1)

			sentences_df['ver_gen'] = 0
			sentences_df['ver_spec'] = 0
			sentences_df = sentences_df[['sentence_id', 'sentence_tuples', 'condition_acid', 'treatment_acid', 'x_train_gen', 'x_train_spec', 'label', 'ver_gen', 'ver_spec']]
			sentences_df.to_sql('training_sentences_with_version', engine, schema='ml2', if_exists='append', index=False, \
				dtype={'sentence_tuples' : sqla.types.JSON, 'x_train_gen' : sqla.types.JSON, 'x_train_spec' : sqla.types.JSON})
		elif write_type == 'spec':
			for i,d in sentences_df.iterrows():
				res = get_labelled_data_sentence_custom_v2(d, d['condition_id'], d['treatment_id'])
				res['label'] = item['label']
				res['ver'] = 0
				res.to_sql('custom_training_sentences', engine, schema='annotation', if_exists='append', \
							index=False, dtype={'sentence_tuples' : sqla.types.JSON, 'x_train_cust' : sqla.types.JSON, 'sentence' : sqla.types.Text})
		cursor.close()
		conn.close()
		engine.dispose()

def get_labelled_data_sentence_generic_v2_custom(sentence, condition_id, tx_word, conditions_set, all_treatments_set):
	conn,cursor = pg.return_postgres_cursor()
	final_results = pd.DataFrame()
	sample = [0]*max_words
	counter = 0

	for index,words in enumerate(sentence['sentence_tuples'][0]):
		if words[1] == condition_id and sample[counter-1] == target_condition_key:
			continue
		elif words[1] == condition_id and sample[counter-1] != target_condition_key:
			sample[counter] = target_condition_key
		elif (words[1] in conditions_set) and (sample[counter-1] != generic_condition_key):
			sample[counter] = generic_condition_key
		elif (words[0] == tx_word) and (sample[counter-1] != target_treatment_key):
			sample[counter] = target_treatment_key
		elif (words[1] in all_treatments_set) and (sample[counter-1] != generic_treatment_key):
			sample[counter] = generic_treatment_key
		elif (words[1] != 0) and (get_word_index(words[1], cursor) != UNK_ind) and (sample[counter-1] != get_word_index(words[1], cursor)):
			sample[counter] = get_word_index(words[1], cursor)
		elif (words[1] == 0) and (get_word_index(words[0], cursor) != UNK_ind) and (sample[counter-1] != get_word_index(words[0], cursor)):
			sample[counter] = get_word_index(words[0], cursor)
		elif ((words[1] == 0) and (get_word_index(words[0], cursor) == UNK_ind)) or ((words[1] == 1) \
			and (get_word_index(words[1], cursor) == UNK_ind)):
			sample[counter] = UNK_ind
		else:
			counter -= 1

		counter += 1

		if counter >= max_words-1:
			break

	cursor.close()
	conn.close()
	return sample

def analyze_sentence(sentence_df, condition_id, cursor):
	# model = load_model('model01.04.19.-01.hdf5')
	model = load_model('double-10.hdf5')

	all_conditions_set = get_all_conditions_set()
	all_treatments_set = get_all_treatments_set()

	final_res = []
	final_res_bounded = []
	for ind,word in enumerate(sentence_df['sentence_tuples'][0]):

		if word[1] == condition_id:
			continue
		else:

			labelled_sentence_gen = get_labelled_data_sentence_generic_v2_custom(sentence_df, condition_id, word[0], \
				all_conditions_set, all_treatments_set)
			labelled_sentence_spec = get_labelled_data_sentence_specific_v2_custom(sentence_df, condition_id, word[0], \
				all_conditions_set, all_treatments_set)
			u.pprint(labelled_sentence_gen)
			u.pprint(labelled_sentence_spec)
			labelled_sentence_gen = np.array([labelled_sentence_gen])
			labelled_sentence_spec = np.array([labelled_sentence_spec])

			res = float(model.predict([labelled_sentence_gen, labelled_sentence_spec]))
			final_res.append((word[0], word[1], res))

	return final_res


def print_contingency(model_name):
	conn, cursor = pg.return_postgres_cursor()
	model = load_model(model_name)

	curr_version = int(pg.return_df_from_query(cursor, "select min(ver_gen) from ml2.test_sentences", \
			None, ['ver_gen'])['ver_gen'][0])
	new_version = curr_version + 1

	# should be OK to load into memory

	testing_query = "select id, sentence_id, x_train_gen, label from ml2.test_sentences where ver_gen=%s"
	sentences_df = pg.return_df_from_query(cursor, testing_query, (curr_version,), \
		['id', 'sentence_id', 'x_train_gen', 'label'])

	zero_zero = 0
	zero_one = 0
	one_zero = 0
	one_one = 0



	for ind,item in sentences_df.iterrows():
		x_train_gen = np.array([item['x_train_gen']])
		res = float(model.predict([x_train_gen])[0][0])

		if ((item['label'] == 1) and (res >= 0.50)):
			one_one += 1
		elif((item['label'] == 1) and (res < 0.50)):
			one_zero += 1
		elif ((item['label'] == 0) and (res < 0.50)):
			zero_zero += 1
		elif ((item['label'] == 0) and (res >= 0.50)):
			zero_one += 1

		# try:
		# 	query = """
		# 		set schema 'annotation';
		# 		UPDATE annotation.test_sentences_v2
		# 		SET ver = %s
		# 		where id = ANY(%s);
		# 	"""
		# 	cursor.execute(query, (new_version, id_list))
		# 	cursor.connection.commit()
		# 	sentences_df = pg.return_df_from_query(cursor, testing_query, (curr_version,), \
		# 		['id', 'sentence_id', 'x_train_gen', 'label'])

		# except:
		# 	print("update version failed. Rolling back")
		# 	cursor.connection.rollback()
		# 	sys.exit(0)

	print(model_name)	
	print("label 1, res 1: " + str(one_one))
	print("label 1, res 0: " + str(one_zero))
	print("label 0, res 0: " + str(zero_zero))
	print("label 0, res 1: " + str(zero_one))

	sens = (one_one) / (one_one + one_zero)
	print("sensitivity : " + str(sens))
	spec = zero_zero / (zero_one + zero_zero)
	print("specificity : " + str(spec))

	cursor.close()
	conn.close()

def build_w2v_embedding():
	conn,cursor = pg.return_postgres_cursor()

	counter = 0
	max_len = "select count(*) as cnt from annotation.training_sentences_with_version_v2"
	max_len = pg.return_df_from_query(cursor, max_len, None, ['cnt'])['cnt'].values
	model = None

	while counter < max_len:
		print(counter)
		sent_query = "select x_train_spec from annotation.training_sentences_with_version_v2 limit 10000 offset %s"
		sentences = pg.return_df_from_query(cursor, sent_query, (counter,), ['x_train_spec'])['x_train_spec'].tolist()
		
		for c1,i in enumerate(sentences):
			for c2,j in enumerate(i):
				sentences[c1][c2]=str(j)
		
		if model == None:
			model = Word2Vec(sentences, size=200, window=5, min_count=1, negative=15, iter=5, workers=4)
		else:
			model.build_vocab(sentences, update=True)
			model.train(sentences, total_examples=1, epochs=5)
		counter += 10000

	model.save('embedding_200.02.20.bin')

if __name__ == "__main__":
	
	# print(get_word_index('195967001', cursor))
	# print(train_data_generator(10, cursor))
	# train_with_word2vec()
	# parallel_treatment_recategorization_top('../model-10.hdf5')
	# parallel_treatment_recategorization_top('../model-10.hdf5')
	# gen_datasets_mp(1)

	# update_word2vec('model.3.hdf5')

	train_with_word2vec()
	# print_contingency('model-01.hdf5')
	# print_contingency('model-02.hdf5')
	# print_contingency('model-03.hdf5')
	# print_contingency('model-04.hdf5')
	# print_contingency('model-05.hdf5')
	# print_contingency('model-06.hdf5')
	# print_contingency('model-07.hdf5')
	# print_contingency('model-08.hdf5')
	# print_contingency('model-09.hdf5')
	# print_contingency('model-10.hdf5')
