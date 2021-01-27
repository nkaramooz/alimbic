import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from nltk.stem.wordnet import WordNetLemmatizer
from operator import itemgetter
import nltk.data
import numpy as np
import time
import utilities.pglib as pg
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

from numpy import array 
from tensorflow.keras import layers
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

import warnings
import plac
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


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






def train_data_generator_v2(batch_size, cursor):

	while True:
		curr_version = int(pg.return_df_from_query(cursor, "select min(ver) from ml2.all_training_sentences", \
			None, ['ver'])['ver'][0])

		new_version = curr_version + 1
		# curr_version_1 = int(pg.return_df_from_query(cursor, "select min(ver) from ml2.all_training_sentences where label=1 and condition_acid in ('10609', '482311', '467', '210918', '125792', '181687') ", \
		# 	None, ['ver'])['ver'][0])
	
		# new_version_1 = curr_version_1 + 1

		# curr_version_0 = int(pg.return_df_from_query(cursor, "select min(ver) from ml2.all_training_sentences where label=0 and condition_acid in ('10609', '482311', '467', '210918', '125792', '181687')", \
		# 	None, ['ver'])['ver'][0])
	
		# new_version_0 = curr_version_0 + 1

		query = "select id, x_train_gen, x_train_mask, label from ml2.all_training_sentences where ver = %s limit %s"
		train_df = pg.return_df_from_query(cursor, query, (curr_version, batch_size), ['id', 'x_train_gen', 'x_train_mask', 'label'])

		# query = "select id, x_train_gen, x_train_mask, label from ml2.all_training_sentences where label=1 and condition_acid in ('10609', '482311', '467', '210918', '125792', '181687') and ver = %s limit %s"
		# train_df_1 = pg.return_df_from_query(cursor, query, (curr_version_1, batch_size/2), ['id', 'x_train_gen', 'x_train_mask', 'label'])

		# query = "select id, x_train_gen, x_train_mask, label from ml2.all_training_sentences where label=0 and and condition_acid in ('10609', '482311', '467', '210918', '125792', '181687') and ver = %s limit %s"
		# train_df_0 = pg.return_df_from_query(cursor, query, (curr_version_0, batch_size/2), ['id', 'x_train_gen', 'x_train_mask', 'label'])

		# train_df = train_df_0.append(train_df_1)

		x_train_gen = train_df['x_train_gen'].tolist()
		x_train_mask = train_df['x_train_mask'].tolist()


		y_train = train_df['label'].tolist()
		

		try:
			id_df = train_df['id'].tolist()
			query = """
				UPDATE ml2.all_training_sentences
				SET ver = %s
				where id = ANY(%s);
			"""
			cursor.execute(query, (new_version, id_df))
			cursor.connection.commit()

			# id_df = train_df_0['id'].tolist()
			# query = """
			# 	UPDATE ml2.all_training_sentences
			# 	SET ver = %s
			# 	where id = ANY(%s);
			# """
			# cursor.execute(query, (new_version_0, id_df))
			# cursor.connection.commit()

			# id_df = train_df_1['id'].tolist()
			# query = """
			# 	UPDATE ml2.all_training_sentences
			# 	SET ver = %s
			# 	where id = ANY(%s);
			# """
			# cursor.execute(query, (new_version_1, id_df))
			# cursor.connection.commit()

			# query = """
			# 	select count(*) as cnt from ml2.all_training_sentences where condition_acid in ('10609', '482311', '467', '210918', '125792', '181687') and label=1 and ver=%s
			# """
			# cnt_1 = int(pg.return_df_from_query(cursor, query, (curr_version_1,), ['cnt'])['cnt'][0])
			# if cnt_1 < 25:
			# 	query = """
			# 		UPDATE ml2.all_training_sentences
			# 		SET ver = %s
			# 		where label=1 and ver=%s;
			# 	"""
			# 	cursor.execute(query, (new_version_1, curr_version_1))
			# 	cursor.connection.commit()

			# query = """
			# 	select count(*) as cnt from ml2.all_training_sentences where condition_acid in ('10609', '482311', '467', '210918', '125792', '181687') and label=0 and ver=%s
			# """
			# cnt_0 = int(pg.return_df_from_query(cursor, query, (curr_version_0,), ['cnt'])['cnt'][0])
			# if cnt_0 < 25:
			# 	query = """
			# 		UPDATE ml2.all_training_sentences
			# 		SET ver = %s
			# 		where label=0 and ver=%s;
			# 	"""
			# 	cursor.execute(query, (new_version_0, curr_version_0))
			# 	cursor.connection.commit()

			# yield [np.asarray(x_train_spec), np.asarray(x_train_mask), np.asarray([x_train_section_ind*max_words])], np.asarray(y_train)
			yield [np.asarray(x_train_gen), np.asarray(x_train_mask)], np.asarray(y_train)
			# print((np.asarray(x_train_gen), np.asarray(y_train)))
			# yield (np.asarray(x_train_gen), np.asarray(y_train))
		except:
			print("update version failed. Rolling back")
			cursor.connection.rollback()
			yield None

	
def train_with_rnn():
	conn,cursor = pg.return_postgres_cursor()

	embedding_size=500
	batch_size = 100
	num_epochs = 20

	model_input = Input(shape=(max_words,))
	model_mask = Input(shape=(max_words,))
	first_model = Embedding(vocabulary_size+1, embedding_size, trainable=True, mask_zero=True)(model_input)
	first_model = Conv1D(filters=32, kernel_size=4, activation='relu')(first_model)
	first_model = Dropout(0.3)(first_model)
	first_model = MaxPooling1D(pool_size=2)(first_model)
	first_model = Flatten()(first_model)
	first_model = layers.concatenate([first_model, model_mask])



	second_model = Embedding(vocabulary_size+1, embedding_size, trainable=True, mask_zero=True)(model_input)
	second_model = Conv1D(filters=32, kernel_size=8, activation='relu')(second_model)
	second_model = Dropout(0.3)(second_model)
	second_model = MaxPooling1D(pool_size=2)(second_model)
	second_model = Flatten()(second_model)
	second_model = layers.concatenate([second_model, model_mask])


	third_model = Embedding(vocabulary_size+1, embedding_size, trainable=True, mask_zero=True)(model_input)
	third_model = Conv1D(filters=32, kernel_size=16, activation='relu')(third_model)
	third_model = Dropout(0.3)(third_model)
	third_model = MaxPooling1D(pool_size=2)(third_model)
	third_model = Flatten()(third_model)
	third_model = layers.concatenate([third_model, model_mask])


	final_model = layers.concatenate([first_model, second_model, third_model])
	final_model = Dense(100, activation='relu', name='merge_dense_100')(final_model)
	final_model = Dense(10, activation='relu', name='merge_dense_10')(final_model)
	pred = Dense(1, activation='sigmoid', name='merge_dense_pred')(final_model)
	

	model = Model(inputs=[model_input, model_mask], outputs=[pred])

	print(model.summary())
	# model.summary(print_fn=lambda x: report.write(x + '\n'))
	# report.write('\n')

	model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

	checkpointer = ModelCheckpoint(filepath='./double-{epoch:02d}.hdf5', verbose=1)

	
	history = model.fit(train_data_generator_v2(batch_size, cursor), \
	 epochs=num_epochs, class_weight={0:1, 1:75}, steps_per_epoch =((4814458//batch_size)+1),
	  callbacks=[checkpointer])



def train_with_word2vec_v2():
	conn,cursor = pg.return_postgres_cursor()

	embedding_size=500
	batch_size = 100
	num_epochs = 20

	first_input = Input(shape=(max_words,))
	second_input = Input(shape=(max_words,))
	# third_input = Input(shape=(1,))

	first_model = Embedding(vocabulary_size+1, embedding_size, trainable=True, mask_zero=True)(first_input)
	second_model = Embedding(vocabulary_size+1, embedding_size, trainable=True, mask_zero=True)(second_input)


	first_model = Bidirectional(LSTM(500, return_sequences=True))(first_model)

	x = layers.concatenate([first_model, second_model])
	y = GlobalMaxPooling1D()(x)
	z = Dense(50, activation='relu', name="dense_1")(y)

	pred = Dense(1, activation="sigmoid", name="prediction")(z)

	model = Model(inputs=[first_input, second_input], outputs=[pred])

	print(model.summary())
	# model.summary(print_fn=lambda x: report.write(x + '\n'))
	# report.write('\n')

	model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

	checkpointer = ModelCheckpoint(filepath='./double-{epoch:02d}.hdf5', verbose=1)

	
	history = model.fit(train_data_generator_v2(batch_size, cursor), \
	 epochs=num_epochs, class_weight={0:1, 1:75}, steps_per_epoch =((4978508//batch_size)+1),
	  callbacks=[checkpointer])


def update_word2vec(model_name):
	conn,cursor = pg.return_postgres_cursor()
	embedding_size=500
	batch_size = 50
	num_epochs = 10
	model = load_model(model_name)

	checkpointer = ModelCheckpoint(filepath='./0106.{epoch:02d}.hdf5', verbose=1)

	# history = model.fit_generator(train_data_generator_v2(batch_size, cursor), \
	#  epochs=num_epochs, class_weight={0:1, 1:50}, steps_per_epoch =((10000//batch_size)+1), callbacks=[checkpointer])
	history = model.fit_generator(train_data_generator_v2(batch_size, cursor), \
	 epochs=num_epochs, steps_per_epoch =((5875//batch_size)+1), callbacks=[checkpointer])

	

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

		# update_query = """
		# 	set schema 'ml2';
		# 	UPDATE treatment_candidates
		# 	SET ver = %s
		# 	where sentence_id = ANY(%s) and condition_acid = ANY(%s) and treatment_acid = ANY(%s);
		# """

		update_query = """
			set schema 'ml2';
			UPDATE treatment_candidates
			SET ver = %s
			where sentence_id = ANY(%s) and treatment_acid = ANY(%s);
		"""

		sentence_id_list = treatment_candidates_df['sentence_id'].tolist()
		condition_acid_list = treatment_candidates_df['condition_acid'].tolist()
		treatment_acid_list = treatment_candidates_df['treatment_acid'].tolist()
		new_version = old_version + 1
		# cursor.execute(update_query, (new_version, sentence_id_list, condition_acid_list, treatment_acid_list))
		cursor.execute(update_query, (new_version, sentence_id_list, treatment_acid_list))
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
	gen = np.array([row['x_train_gen']])
	mask = np.array([row['x_train_mask']])


	res = float(model.predict([gen, mask])[0][0])
	return res


def batch_treatment_recategorization(model_name, treatment_candidates_df, all_conditions_set, all_treatments_set):
	model = load_model(model_name)
	conn,cursor = pg.return_postgres_cursor()
	engine = pg.return_sql_alchemy_engine()
	conditions_set = get_all_conditions_set()
	treatments_set = get_all_treatments_set()

	treatment_candidates_df = treatment_candidates_df.apply(apply_get_generic_labelled_data, \
	 all_conditions_set=all_conditions_set, all_treatments_set=all_treatments_set, axis=1)

	treatment_candidates_df['score'] = treatment_candidates_df.apply(apply_score, model=model, axis=1)
	treatment_candidates_df.to_sql('treatment_recs_staging', engine, schema='ml2', if_exists='append', \
	 index=False, dtype={'sentence_tuples' : sqla.types.JSON})

	cursor.close()
	conn.close()
	engine.dispose()

mask_condition_marker = 1
mask_treatment_marker = 2

def get_labelled_data_sentence_specific_v2(sentence, condition_id, tx_id):
	conn,cursor = pg.return_postgres_cursor()
	final_results = pd.DataFrame()
	sample = [0]*max_words 
	mask = [0]*max_words
	counter = 0
	
	for index,words in enumerate(sentence['sentence_tuples']):
		if words[1] == condition_id and sample[counter-1] == get_word_index(words[1], cursor):
			continue
		elif words[1] == condition_id and sample[counter-1] != get_word_index(words[1], cursor):
			sample[counter] = get_word_index(words[1], cursor)
			mask[counter] = mask_condition_marker
		elif (words[1] == tx_id) and (sample[counter-1] == get_word_index(words[1], cursor)):
			continue
		elif (words[1] == tx_id) and (sample[counter-1] != get_word_index(words[1], cursor)):
			sample[counter] = get_word_index(words[1], cursor)
			mask[counter] = mask_treatment_marker
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
	sentence['x_train_spec'] = sample
	sentence['x_train_mask'] = mask
	return sentence

# sample[0] = dict value of item
# sample[1]: 1=target_condition, 2=target_treatment
def get_labelled_data_sentence_generic_v2(sentence, conditions_set, all_treatments_set):
	conn,cursor = pg.return_postgres_cursor()
	condition_id = sentence['condition_acid']
	tx_id = sentence['treatment_acid']

	final_results = pd.DataFrame()
	sample = [0]*max_words
	mask = [0]*max_words
	counter = 0

	for index,words in enumerate(sentence['sentence_tuples']):		
		if words[1] == condition_id and sample[counter-1] == target_condition_key:
			continue
		elif words[1] == condition_id and sample[counter-1] != target_condition_key:
			sample[counter] = target_condition_key
			mask[counter] = mask_condition_marker
		elif (words[1] in conditions_set) and (sample[counter-1] == generic_condition_key):
			continue
		elif (words[1] in conditions_set) and (sample[counter-1] != generic_condition_key):
			sample[counter] = generic_condition_key
		elif (words[1] == tx_id and (sample[counter-1] == target_treatment_key)):
			continue
		elif (words[1] == tx_id) and (sample[counter-1] != target_treatment_key):
			sample[counter] = target_treatment_key
			mask[counter] = mask_treatment_marker
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
	sentence['x_train_gen'] = sample
	sentence['x_train_mask'] = mask
	return sentence


def get_labelled_data_sentence_generic_v2_custom(sentence, condition_id, tx_word, conditions_set, all_treatments_set):
	conn,cursor = pg.return_postgres_cursor()
	final_results = pd.DataFrame()
	sample = [0]*max_words
	mask = [0]*max_words
	counter = 0

	for index,words in enumerate(sentence['sentence_tuples'][0]):
		if words[1] == condition_id and sample[counter-1] == target_condition_key:
			continue
		elif words[1] == condition_id and sample[counter-1] != target_condition_key:
			sample[counter] = target_condition_key
			mask[counter] = mask_condition_marker
		elif (words[1] in conditions_set) and (sample[counter-1] == generic_condition_key):
			continue
		elif (words[1] in conditions_set) and (sample[counter-1] != generic_condition_key):
			sample[counter] = generic_condition_key
		elif (words[0] == tx_word) and (sample[counter-1] != target_treatment_key):
			sample[counter] = target_treatment_key
			mask[counter] = mask_treatment_marker
		elif (words[1] in all_treatments_set) and (sample[counter-1] == generic_treatment_key):
			continue
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
	return sample, mask



# Below is used for sentence not annotated with concepts
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
						from pubmed.sentence_concept_arr_1_8 t1
						join (select root_acid as condition_acid from annotation2.base_concept_types where rel_type='condition' or rel_type='symptom') t2
							on t2.condition_acid = ANY(t1.concept_arr::text[])
						join (select acid as treatment_acid from annotation2.downstream_root_cid where acid in %s) t3
							on t3.treatment_acid = ANY(t1.concept_arr::text[])
						join pubmed.sentence_tuples_1_8 t4
							on t1.sentence_id = t4.sentence_id
					"""
		sentences_df = pg.return_df_from_query(cursor, sentences_query, (tuple(tx_id_arr),), \
				['sentence_tuples', 'condition_acid', 'treatment_acid', 'sentence_id', 'section_ind', 'pmid'])
			
		print("finished query")
			
		print("len of wildcard: " + str(len(sentences_df)))
	elif treatment_acid != '%':
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
				from pubmed.sentence_concept_arr_1_8 t1
				join (select acid as condition_acid from annotation2.downstream_root_cid where acid in %s) t2
					on t2.condition_acid = ANY(t1.concept_arr::text[])
				join (select acid as treatment_acid from annotation2.downstream_root_cid where acid in %s) t3
					on t3.treatment_acid = ANY(t1.concept_arr::text[])
				join pubmed.sentence_tuples_1_8 t4
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
	number_of_processes = 30
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
			sentences_df = sentences_df.apply(apply_get_generic_labelled_data, \
				all_conditions_set=conditions_set, all_treatments_set=treatments_set, axis=1)
			# sentences_df['x_train_spec'] = sentences_df.apply(apply_get_specific_labelled_data, axis=1)

			sentences_df['ver'] = 0
			sentences_df = sentences_df[['sentence_id', 'sentence_tuples', 'condition_acid', 'treatment_acid',\
			 'x_train_gen', 'x_train_mask', 'label', 'ver']]
			sentences_df.to_sql('training_sentences_with_version', engine, schema='ml2', if_exists='append', index=False, \
				dtype={'sentence_tuples' : sqla.types.JSON, 'x_train_gen' : sqla.types.JSON, 'x_train_mask' : sqla.types.JSON, 'sentence' : sqla.types.Text})
		
		elif write_type == 'spec':
			sentences_df = sentences_df.apply(apply_get_specific_labelled_data, axis=1)
			sentences_df['ver'] = 0
			sentences_df.to_sql('custom_training_sentences', engine, schema='ml2', if_exists='append', \
						index=False, dtype={'sentence_tuples' : sqla.types.JSON, 'x_train_spec' : sqla.types.JSON, 'x_train_mask' : sqla.types.JSON, 'sentence' : sqla.types.Text})
		
		cursor.close()
		conn.close()
		engine.dispose()

def analyze_sentence(model_name, sentence, condition_id):

	term = ann2.clean_text(sentence)
	all_words = ann2.get_all_words_list(term)
	cache = ann2.get_cache(all_words, False)
	item = pd.DataFrame([[term, 'title', 0, 0]], columns=['line', 'section', 'section_ind', 'ln_num'])
	sentence_annotations_df, sentence_tuples_df, sentence_concept_arr_df = ann2.annotate_text_not_parallel(item, cache, True, True, True)
	print(sentence_tuples_df)
	model = load_model(model_name)

	all_conditions_set = get_all_conditions_set()
	all_treatments_set = get_all_treatments_set()

	

	final_res = []

	for ind,word in enumerate(sentence_tuples_df['sentence_tuples'][0]):
		if word[1] == condition_id:
			continue
		else:

			sample, mask = get_labelled_data_sentence_generic_v2_custom(sentence_tuples_df, condition_id, word[0], \
				all_conditions_set, all_treatments_set)
			sample_arr = np.array([sample])
			mask_arr = np.array([mask])

			res = float(model.predict([sample_arr, mask_arr]))
			final_res.append((word[0], word[1], res))

	print(final_res)
	return final_res


def print_contingency(model_name):
	conn, cursor = pg.return_postgres_cursor()
	model = load_model(model_name)

	curr_version = int(pg.return_df_from_query(cursor, "select min(ver) from ml2.test_sentences_subset", \
			None, ['ver'])['ver'][0])
	new_version = curr_version + 1

	# should be OK to load into memory

	testing_query = "select id, sentence_id, x_train_gen, x_train_mask, label from ml2.test_sentences_subset where ver=%s"
	sentences_df = pg.return_df_from_query(cursor, testing_query, (curr_version,), \
		['id', 'sentence_id', 'x_train_gen','x_train_mask', 'label'])

	zero_zero = 0
	zero_one = 0
	one_zero = 0
	one_one = 0



	for ind,item in sentences_df.iterrows():
		x_train_spec = np.array([item['x_train_gen']])
		x_train_mask = np.array([item['x_train_mask']])

		res = float(model.predict([x_train_spec, x_train_mask])[0][0])

		if ((item['label'] == 1) and (res >= 0.50)):
			one_one += 1
		elif((item['label'] == 1) and (res < 0.50)):
			one_zero += 1
		elif ((item['label'] == 0) and (res < 0.50)):
			zero_zero += 1
		elif ((item['label'] == 0) and (res >= 0.50)):
			zero_one += 1


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



def gen_spacy_dataset_worker(input, conditions_set, treatments_set):
	for func,args in iter(input.get, 'STOP'):
		gen_spacy_dataset_calculate(func, args, conditions_set, treatments_set)

def gen_spacy_dataset_calculate(func, args, conditions_set, treatments_set):
	func(*args, conditions_set, treatments_set)

def gen_spacy_dataset_mp(new_version):
	number_of_processes = 30
	old_version = new_version-1
	conditions_set = get_all_conditions_set()
	treatments_set = get_all_treatments_set()

	task_queue = mp.Queue()
	pool = []

	for i in range(number_of_processes):
		p = mp.Process(target=gen_spacy_dataset_worker, args=(task_queue, conditions_set, treatments_set))
		pool.append(p)
		p.start()

	conn,cursor = pg.return_postgres_cursor()

	query = """
			select 
				t1.id
				,condition_acid::text
				,treatment_acid::text
				,label
				,ver 
			from ml2.labelled_treatments t1 
			join annotation2.concept_types t2
			on t1.treatment_acid = t2.root_acid 
			where ver=%s and label=0 and t2.active=1 and t2.rel_type='treatment' limit %s
		"""
	labels_df = pg.return_df_from_query(cursor, query, (old_version, number_of_processes), ['id', 'condition_acid', 'treatment_acid', 'label', 'ver'])
	

	while len(labels_df.index) > 0:
		for index,item in labels_df.iterrows():
			params = (item['condition_acid'], item['treatment_acid'], item['label'])
			task_queue.put((gen_spacy_datasets_mp_bottom, params))

		query = "UPDATE ml2.labelled_treatments set ver = %s where id in %s"
		cursor.execute(query, (new_version, tuple(labels_df['id'].values.tolist())))
		cursor.connection.commit()

		query = """
			select 
				t1.id
				,condition_acid::text
				,treatment_acid::text
				,label
				,ver 
			from ml2.labelled_treatments t1 
			join annotation2.concept_types t2
			on t1.treatment_acid = t2.root_acid 
			where ver=%s and label=0 and t2.active=1 and t2.rel_type='treatment' limit %s
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

def gen_spacy_datasets_mp_bottom(condition_acid, treatment_acid, label, conditions_set, treatments_set):
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
						from pubmed.sentence_concept_arr_1_8 t1
						join (select root_acid as condition_acid from annotation2.base_concept_types where rel_type='condition' or rel_type='symptom') t2
							on t2.condition_acid = ANY(t1.concept_arr::text[])
						join (select acid as treatment_acid from annotation2.downstream_root_cid where acid in %s) t3
							on t3.treatment_acid = ANY(t1.concept_arr::text[])
						join pubmed.sentence_tuples_1_8 t4
							on t1.sentence_id = t4.sentence_id
					"""
		sentences_df = pg.return_df_from_query(cursor, sentences_query, (tuple(tx_id_arr),), \
				['sentence_tuples', 'condition_acid', 'treatment_acid', 'sentence_id', 'section_ind', 'pmid'])
			
		print("finished query")
			
		print("len of wildcard: " + str(len(sentences_df)))
	elif treatment_acid != '%':
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
				from pubmed.sentence_concept_arr_1_8 t1
				join (select acid as condition_acid from annotation2.downstream_root_cid where acid in %s) t2
					on t2.condition_acid = ANY(t1.concept_arr::text[])
				join (select acid as treatment_acid from annotation2.downstream_root_cid where acid in %s) t3
					on t3.treatment_acid = ANY(t1.concept_arr::text[])
				join pubmed.sentence_tuples_1_8 t4
					on t1.sentence_id = t4.sentence_id
			"""
		sentences_df = pg.return_df_from_query(cursor, sentences_query, (tuple(condition_id_arr), tuple(tx_id_arr)), \
				['sentence_tuples', 'condition_acid', 'treatment_acid', 'sentence_id', 'section_ind', 'pmid'])
	cursor.close()
	conn.close()

	if len(sentences_df.index) > 0:
		sentences_df['label'] = label
		write_spacy_sentence_vectors_from_labels(sentences_df, conditions_set, treatments_set, 'gen')


def write_spacy_sentence_vectors_from_labels(sentences_df, conditions_set, treatments_set, write_type):
	conn,cursor = pg.return_postgres_cursor()
	engine = pg.return_sql_alchemy_engine()

	if len(sentences_df.index) > 0:
		# sentences_df = sentences_df.apply(apply_spacy_vectors, \
		# 		all_conditions_set=conditions_set, all_treatments_set=treatments_set, axis=1)
		sentences_df['ver'] = 0
		sentences_df = sentences_df[['sentence_id', 'sentence_tuples', 'condition_acid', 'treatment_acid', 'label', 'ver']]
		sentences_df.to_sql('spacy_training_data', engine, schema='ml2', if_exists='append', index=False, \
				dtype={'sentence_tuples' : sqla.types.JSON, 'sentence' : sqla.types.Text})

		cursor.close()
		conn.close()
		engine.dispose()

def apply_spacy_vectors(row, all_conditions_set, all_treatments_set):
	return get_spacy_vectors(row, all_conditions_set, all_treatments_set)

def get_spacy_vectors(sentence, conditions_set, all_treatments_set):
	conn,cursor = pg.return_postgres_cursor()
	condition_id = sentence['condition_acid']
	tx_id = sentence['treatment_acid']
	label = sentence['label']

	char_annotation = []
	reconstructed_sentence = ""
	char_index = 0
	for index,words in enumerate(sentence['sentence_tuples']):
		reconstructed_sentence += words[0] + ' '
		if words[1] == condition_id and (len(char_annotation)> 0) and char_annotation[len(char_annotation)-1][2] == 'COND_IND':
			target_tuple = char_annotation[len(char_annotation)-1]
			char_annotation[len(char_annotation)-1] = (target_tuple[0], char_index+len(words[0]), 'COND_IND')

		elif words[1] == condition_id:
			char_annotation.append((char_index, char_index+len(words[0]), 'COND_IND'))		

		elif words[1] == tx_id:
			if label == 1 and (len(char_annotation)>0) and char_annotation[len(char_annotation)-1][2] == 'TX_COND':
				target_tuple = char_annotation[len(char_annotation)-1]
				char_annotation[len(char_annotation)-1] = (target_tuple[0], char_index+len(words[0]), 'TX_COND')
			elif label == 1 and words[1] == tx_id:
				char_annotation.append((char_index, char_index+len(words[0]), 'TX_COND'))
			elif label == 0 and (len(char_annotation)>0) and char_annotation[len(char_annotation)-1][2] == 'NOT_TX':
				target_tuple = char_annotation[len(char_annotation)-1]
				char_annotation[len(char_annotation)-1] = (target_tuple[0], char_index+len(words[0]), 'NOT_TX')
			elif label == 0 and words[1] == tx_id:
				char_annotation.append((char_index, char_index+len(words[0]), 'NOT_TX'))

		char_index += len(words[0]) + 1

	sentence['sentence'] = reconstructed_sentence
	sentence['spacy_label'] = char_annotation
	cursor.close()
	conn.close()
	return sentence

def get_spacy_training_data(batch_size):
	conn,cursor = pg.return_postgres_cursor()

	curr_version = int(pg.return_df_from_query(cursor, "select min(ver) from ml2.spacy_training_data", \
			None, ['ver'])['ver'][0])

		# Note that ordering by random adds significant time
	new_version = curr_version + 1
	query = "select sentence_id, sentence, spacy_label from ml2.spacy_training_data where ver = %s limit %s"
	train_df = pg.return_df_from_query(cursor, query, (curr_version, batch_size), ['id', 'sentence', 'spacy_label'])

	# sentence_list = train_df['sentence'].tolist()
	# label_list = train_df['spacy_label'].tolist()
	print("STOP")
	print(train_df)
	train_data = []
	for index,item in train_df.iterrows():
		train_data.append((item['sentence'], {"entities" : item['spacy_label']}))

	id_df = train_df['id'].tolist()
	conn.close()
	cursor.close()
	return train_data
	# try:
		# query = """
		# 	UPDATE ml2.spacy_training_data
		# 	SET ver = %s
		# 	where id = ANY(%s);
		# """
		# cursor.execute(query, (new_version, id_df))
		# cursor.connection.commit()

			# yield [np.asarray(x_train_spec), np.asarray(x_train_mask), np.asarray([x_train_section_ind*max_words])], np.asarray(y_train)
		
			# print((np.asarray(x_train_gen), np.asarray(y_train)))
			# yield (np.asarray(x_train_gen), np.asarray(y_train))
	# except:
	# 	print("update version failed. Rolling back")
	# 	cursor.connection.rollback()
	# 	yield None





if __name__ == "__main__":
	# plac.call(main)


	# nlp = spacy.load('/home/nkaramooz/Documents/alimbic')

	# doc2 = nlp("Levofloxacin for the treatment of myocardial infarction")
	# print(doc2)
	# for ent in doc2.ents:
	# 	print(ent)
	# 	print(ent.label_, ent.text)
	# sentence = "mRNA vaccine for COVID-19"
	# sentence = "amiodarone for patients with covid-19"
	sentence = "amiodarone improves symptoms in patients with covid-19"
	# sentence = "neutralizing antibody therapy in covid-19"
	# sentence = "a patient with cough and bloody diarrhea in covid-19"
	# sentence = sentence.lower()
	model_name = 'double-11.hdf5'
	# model_name = 'alice1.hdf5'
	condition_id = '125792'
	analyze_sentence(model_name, sentence, condition_id)



	# parallel_treatment_recategorization_top('../double-19.hdf5')
	# parallel_treatment_recategorization_top('0106.10.hdf5')


	# gen_datasets_mp(1)

	# update_word2vec('0106.10.hdf5')

	# V2 uses the specific + mask
	# train_with_word2vec_v2()

	# train_with_rnn()
	# print_contingency('double-01.hdf5')
	# print_contingency('double-02.hdf5')
	# print_contingency('double-03.hdf5')
	# print_contingency('double-04.hdf5')
	# print_contingency('double-05.hdf5')
	# print_contingency('double-06.hdf5')
	# print_contingency('double-07.hdf5')
	# print_contingency('double-08.hdf5')
	# print_contingency('double-09.hdf5')
	# print_contingency('double-10.hdf5')
	# print_contingency('double-11.hdf5')
	# print_contingency('double-12.hdf5')
	# print_contingency('double-18.hdf5')
	# print_contingency('double-19.hdf5')
	# print_contingency('double-20.hdf5')
	# print_contingency('model12.0601.hdf5')
	# print_contingency('model12.0602.hdf5')
	# print_contingency('model12.0603.hdf5')