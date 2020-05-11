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
import snomed_annotator as ann
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
# UNK should have index of 1, but unable to change this now
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
	query = "select root_cid from annotation.concept_types where rel_type='condition'"
	conn,cursor = pg.return_postgres_cursor()
	all_conditions_set = set(pg.return_df_from_query(cursor, query, None, ['root_cid'])['root_cid'].tolist())
	return all_conditions_set

def get_all_treatments_set():
	
	query = "select root_cid from annotation.concept_types where rel_type='treatment'"
	conn,cursor = pg.return_postgres_cursor()
	all_treatments_set = set(pg.return_df_from_query(cursor, query, None, ['root_cid'])['root_cid'].tolist())
	return all_treatments_set

def load_word_counts():
	conn,cursor = pg.return_postgres_cursor()
	all_conditions_set = get_all_conditions_set()

	query = "select id, sentence_tuples, version from annotation.distinct_sentences t1 where t1.version = 0 limit 10000"
	sentence_df = pg.return_df_from_query(cursor, query, None, ['id', 'sentence_tuples', 'version'])
	new_version = int(sentence_df['version'].values[0]) + 1

	while len(sentence_df.index) > 0:
		sentence_array = sentence_df['sentence_tuples'].tolist()
		apply_word_counts_to_dict(cursor, sentence_array)
		update_query = "UPDATE annotation.distinct_sentences set version = %s where id in %s"
		cursor.execute(update_query, (new_version, tuple(sentence_df['id'].tolist())))
		cursor.connection.commit()
		sentence_df = pg.return_df_from_query(cursor, query, None, ['id', 'sentence_tuples', 'version'])


# {word : count} in current batch
## need to update postgres table with 

# tmp counts
# select union sum word
# drop tmp count

def apply_word_counts_to_dict(cursor, sentence_row):
	last_word = None
	for index,words in enumerate(sentence_row[0]):
		word_key = None
		if words[1] != 0 and last_word != words[1]:
			# if words[1] in all_conditions_set:
				# update_counts(counts, generic_condition_key)
				# word_key = generic_condition_key
			# else:
				# update_counts(counts, words[1])
			word_key = words[1]
			last_word = words[1]
		elif words[1] == 0:
			word_key = words[0]
			last_word = words[0]

		if word_key is not None:
			try:
				query = """
					set schema 'annotation';
					INSERT INTO word_counts (id, word, count) VALUES (public.uuid_generate_v4(), %s, 1) ON CONFLICT (word) DO UPDATE SET count = word_counts.count + 1"""
				cursor.execute(query, (word_key,))
				cursor.connection.commit()
			except:
				cursor.connection.rollback()


def get_word_index(word, cursor):
	query = "select rn from annotation.word_counts_50k where word=%s limit 1"
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
		query = "select word from annotation.word_counts_50k where rn = %s limit 1"
		word_df = pg.return_df_from_query(cursor, query, (index,), ['word'])
		return str(word_df['word'][0])


def train_data_generator(batch_size, cursor):

	while True:
		curr_version = int(pg.return_df_from_query(cursor, "select min(ver_gen) from annotation.training_sentences", \
			None, ['ver'])['ver'][0])

		query = "select id, x_train_gen, label from annotation.training_sentences where ver_gen=%s order by random() limit %s"
		train_df = pg.return_df_from_query(cursor, query, (curr_version, batch_size), ['id', 'x_train_gen', 'label'])

		x_train_gen = train_df['x_train_gen'].tolist()
		# x_train_spec = train_df['x_train_spec'].tolist()
		y_train = train_df['label'].tolist()
		id_df = train_df['id'].tolist()

		try:
		
			new_version = curr_version + 1
			query = """
				set schema 'annotation';
				UPDATE annotation.training_sentences
				SET ver_gen = %s
				where id in %s
			"""
			cursor.execute(query, (new_version, tuple(id_df)))
			cursor.connection.commit()
		
			# yield [np.asarray(x_train_gen), np.asarray(x_train_spec)], np.asarray(y_train)
			
			yield (np.asarray(x_train_gen), np.asarray(y_train))
		except:
			print("update version failed. Rolling back")
			cursor.connection.rollback()
			yield None

	
## 0 label = 2810596
## 1 label = 62767
def train_with_word2vec():
	conn,cursor = pg.return_postgres_cursor()
	report = open('ml_report.txt', 'w')
	embedding_size=500
	batch_size = 500
	num_epochs = 4

	model=Sequential()
	model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words, trainable=True))
	model.add(LSTM(500, return_sequences=True, input_shape=(embedding_size, batch_size)))
	model.add(Dropout(0.3))
	model.add(TimeDistributed(Dense(500)))
	model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
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
	 epochs=num_epochs, class_weight={0:1, 1:5}, steps_per_epoch =((3165664//batch_size)+1),
	 callbacks=[checkpointer])

	report.close()

def train_with_word2vec_v2():
	conn,cursor = pg.return_postgres_cursor()
	# report = open('ml_report.txt', 'w')
	embedding_size=100
	batch_size = 500
	num_epochs = 10

	model=Sequential()
	model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words, trainable=True))
	model.add(LSTM(500, return_sequences=True, input_shape=(embedding_size, batch_size)))
	# model.add(LSTM(500, return_sequences=True))
	# model.add(LSTM(400, return_sequences=True))
	model.add(Dropout(0.3))
	model.add(TimeDistributed(Dense(500)))
	# model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))


	# inp1 = Input(shape=(max_words,))
	# first_model = Embedding(vocabulary_size, embedding_size, input_length=max_words, trainable=True)(inp1)
	# first_model = Bidirectional(LSTM(128, dropout=0.7, recurrent_dropout=0.7, return_sequences=True, input_shape=(embedding_size, batch_size)))(first_model)
	# first_model = TimeDistributed(Dense(50))(first_model)
	# first_model = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(first_model)
	# first_model = Dropout(0.3)(first_model)
	# first_model = Dense(50, activation='relu')(first_model)
	# first_model = Dropout(0.3)(first_model)
	# first_model = Dense(10, activation='relu')(first_model)
	# first_model = Flatten()(first_model)
	# first_model = Dense(1, activation='sigmoid')(first_model)
	# first_model = Model(inputs=[inp1], outputs=first_model)


	# inp1 = Input(shape=(max_words,))
	# first_model = Embedding(vocabulary_size, embedding_size, input_length=max_words, trainable=True)(inp1)
	# first_model = LSTM(50, return_sequences=True, input_shape=(embedding_size, batch_size))(first_model)

	# first_model = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(first_model)
	# first_model = Dropout(0.3)(first_model)
	# first_model = Dense(25, activation='relu')(first_model)



	# inp2 = Input(shape=(max_words,))
	# second_model = Embedding(vocabulary_size, embedding_size, input_length=max_words, trainable=True)(inp2)
	# second_model = LSTM(50, return_sequences=True, input_shape=(embedding_size, batch_size))(second_model)

	# second_model = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(second_model)
	# second_model = Dropout(0.3)(second_model)
	# second_model = Dense(25, activation='relu')(second_model)



	# c = Concatenate()(inputs=[first_model, second_model])
	# out = Embedding(vocabulary_size, embedding_size, trainable=True)(c)
	# out = Bidirectional(LSTM(100))(c)
	# out = Dense(25, activation='relu')(c)
	# out = Dropout(0.3)(out)
	# out = Dense(10, activation='relu')(out)
	# out = Flatten()(out)
	# out = Dense(1, activation='sigmoid')(out)
	# model = Model(inputs=[inp1, inp2], outputs=out)

	print(model.summary())
	# model.summary(print_fn=lambda x: report.write(x + '\n'))
	# report.write('\n')

	model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

	checkpointer = ModelCheckpoint(filepath='./double-{epoch:02d}.hdf5', verbose=1)

	# try decreasing class weigth to 20
	history = model.fit_generator(train_data_generator(batch_size, cursor), \
	 epochs=num_epochs, class_weight={0:1, 1:20}, steps_per_epoch =((3620302//batch_size)+1),
	  callbacks=[checkpointer])


	# inputA = Input(shape=(max_words,500))
	# inputB = Input(shape=(max_words,, 500))

	# x = Dense(8, activation="relu")(inputA)
	# x = Dense(4, activation="relu")(x)
	# x = Model(inputs=inputA, outputs=x)

	# y = Dense(8, activation="relu")(inputB)
	# y = Dense(4, activation="relu")(y)
	# y = Model(inputs=inputB, outputs=y)

	# print(x.output)
	# print(y.output)
	
	# combined = Concatenate(axis=-1)([x.output, y.output])
	
	# z = Dense(2, activation="relu")(combined)
	# z = Dense(1, activation="linear")(z)
	
	# model = Model(inputs=train_data_generator(batch_size, cursor), outputs=z)
	# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# history = model.fit_generator(train_data_generator(batch_size, cursor), epochs=num_epochs,
	# 	class_weight={0:1, 1:5}, steps_per_epoch =((2884713//batch_size)+1), callbacks=[checkpointer])

	# main_input = Input(shape=(max_words,), dtype='int32', name='main_input')
	# x = Embedding(output_dim=512, input_dim=vocabulary_size+3, input_length=max_words, trainable=True)(main_input)

	# input is 60,2
	# x = Embedding(vocabulary_size, embedding_size, input_length=max_words, trainable=True)
	# lstm_out = LSTM(500)(x)
	# auxilliary_input = Input(shape=(max_words,), name='aux_input')
	# x = Concatenate(axis=-1)([lstm_out, auxilliary_input])
	# x = Dense(64, activation='relu')(x)
	# main_output=Dense(1, activation='sigmoid', name='main_output')(x)
	# model = Model(inputs=train_data_generator(batch_size,cursor), outputs=[main_output])
	# model.compile(loss='binary_crossentropy', 
 #             optimizer='adam', 
 #             metrics=['accuracy'])
	# history = model.fit_generator(train_data_generator(batch_size, cursor), epochs=num_epochs,
	# 	class_weight={0:1, 1:5}, steps_per_epoch =((2884713//batch_size)+1), callbacks=[checkpointer])

	# model=Sequential()
	# model.add(Input(shape=(max_words, 2), dtype='int32'))
	# model.add(Embedding(vocabulary_size+3, embedding_size, input_length=max_words, trainable=True))
	# model.add(LSTM(500, return_sequences=True, input_shape=(embedding_size, batch_size)))
	# model.add(Dropout(0.3))
	# model.add(TimeDistributed(Dense(500)))
	# model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
	# model.add(Flatten())
	# model.add(Dense(1, activation='sigmoid'))

	# print(model.summary())
	# model.summary(print_fn=lambda x: report.write(x + '\n'))
	# report.write('\n')

	# model.compile(loss='binary_crossentropy', 
 #             optimizer='adam', 
 #             metrics=['accuracy'])
	# checkpointer = ModelCheckpoint(filepath='./model-{epoch:02d}.hdf5', verbose=1)

	# history = model.fit_generator(train_data_generator(batch_size, cursor), \
	#  epochs=num_epochs, class_weight={0:1, 1:5}, steps_per_epoch =((2884713//batch_size)+1), callbacks=[checkpointer])

	# report.close()

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

	max_query = "select count(*) as cnt from annotation.title_treatment_candidates_filtered_final"
	max_counter = pg.return_df_from_query(cursor, max_query, None, ['cnt'])['cnt'].values[0]

	counter = 0
	while counter < max_counter:
		parallel_treatment_recategorization_bottom(model_name, counter, all_conditions_set, all_treatments_set, max_counter)
		counter += 16000


def parallel_treatment_recategorization_bottom(model_name, start_row, all_conditions_set, all_treatments_set, max_counter):
	conn,cursor = pg.return_postgres_cursor()
	number_of_processes = 8
	engine_arr = []

	for i in range(number_of_processes):
		engine = pg.return_sql_alchemy_engine()
		engine_arr.append(engine)

	task_queue = mp.Queue()

	pool = []
	for i in range(number_of_processes):
		p = mp.Process(target=recat_worker, args=(task_queue, engine_arr[i]))
		pool.append(p)
		p.start()

	counter = start_row

	while (counter <= start_row+16000) and (counter <= max_counter):
		u.pprint(counter)
		treatment_candidates_query = """select sentence, sentence_tuples, condition_id, 
			treatment_id, pmid, id, section from annotation.title_treatment_candidates_filtered_final 
			limit 2000 offset %s"""
		treatment_candidates_df = pg.return_df_from_query(cursor, treatment_candidates_query, \
			(counter,), ['sentence', 'sentence_tuples', 'condition_id', 'treatment_id', 'pmid', 'id', 'section'])

		params = (model_name, treatment_candidates_df, all_conditions_set, all_treatments_set)
		task_queue.put((batch_treatment_recategorization, params))
		counter += 2000

	for i in range(number_of_processes):
		task_queue.put('STOP')

	for p in pool:
		p.join()

	for g in engine_arr:
		g.dispose()


def recat_worker(input, engine):
	for func,args in iter(input.get, 'STOP'):
		recat_calculate(func, args, engine)


def recat_calculate(func, args, engine):
	func(*args, engine)


def apply_get_generic_labelled_data(row, all_conditions_set, all_treatments_set):
	return get_labelled_data_sentence_generic_v2(row, row['condition_id'], row['treatment_id'], all_conditions_set, all_treatments_set)

def apply_get_specific_labelled_data(row):
	return get_labelled_data_sentence_specific_v2(row, row['condition_id'], row['treatment_id'])

def apply_score(row, model):
	# print([row['x_train_gen'], row['x_train_spec']])
	gen = np.array([row['x_train_gen']])
	spec = np.array([row['x_train_spec']])

	res = float(model.predict([gen, spec])[0][0])
	return res


def batch_treatment_recategorization(model_name, treatment_candidates_df, all_conditions_set, all_treatments_set, engine):
	model = load_model(model_name)
	treatment_candidates_df['x_train_gen'] = treatment_candidates_df.apply(apply_get_generic_labelled_data, \
		all_conditions_set=all_conditions_set, all_treatments_set=all_treatments_set, axis=1)
	treatment_candidates_df['x_train_spec'] = treatment_candidates_df.apply(apply_get_specific_labelled_data, axis=1)
	treatment_candidates_df['score'] = treatment_candidates_df.apply(apply_score, model=model, axis=1)
	treatment_candidates_df.to_sql('raw_treatment_recs_staging_5', engine, schema='annotation', if_exists='append', index=False, dtype={'sentence_tuples' : sqla.types.JSON, 'concept_arr' : sqla.types.JSON})
	u.pprint("write")


# Uses conceptid for treatments
# def get_labelled_data_sentence(sentence, condition_id, tx_id, conditions_set, all_treatments_set):
# 	conn,cursor = pg.return_postgres_cursor()
# 	final_results = pd.DataFrame()

# 	sample = [0]*max_words
	
# 	counter=0
# 	for index,words in enumerate(sentence['sentence_tuples']):
# 		## words[1] is the conceptid, words[0] is the word
# 		# condition 1 -- word is the condition of interest

# 		if words[1] == condition_id and sample[counter-1] == target_condition_key:
# 			continue
# 		elif words[1] == condition_id and sample[counter-1] != target_condition_key:
# 			sample[counter] = target_condition_key
# 		elif (words[1] in conditions_set) and (sample[counter-1] != generic_condition_key):
# 			sample[counter] = generic_condition_key
# 			# - word is generic treatment
# 		elif (words[1] == tx_id) and (sample[counter-1] != target_treatment_key):
# 			sample[counter] = target_treatment_key
# 		elif (words[1] in all_treatments_set) and (sample[counter-1] != generic_treatment_key):
# 			sample[counter] = generic_treatment_key
# 			# Now using conceptid if available
# 		elif words[1] != 0 and (get_word_index(words[1], cursor) != UNK_ind) and get_word_index(words[1], cursor) != sample[counter-1]:
# 			sample[counter] = get_word_index(words[1], cursor)
# 		elif (words[1] == 0 and (get_word_index(words[0], cursor) != UNK_ind) and get_word_index(words[0], cursor) != sample[counter-1])\
# 			and words[1] not in conditions_set and words[1] not in all_treatments_set:
# 			sample[counter] = get_word_index(words[0], cursor)
# 		elif (words[1] == 0) and (get_word_index(words[0], cursor) == UNK_ind) and \
# 			words[1] not in conditions_set and words[1] not in all_treatments_set:
# 			sample[counter] = get_word_index(words[0], cursor)
# 		else:
# 			counter -= 1
		
# 		counter += 1

# 		if counter >= max_words-1:
# 			break

# 	cursor.close()
# 	conn.close()
# 	return sample

# sample[0] = dict value of item
# sample[1]: 1=target_condition, 2=target_treatment
def get_labelled_data_sentence_generic_v2(sentence, condition_id, tx_id, conditions_set, all_treatments_set):
	conn,cursor = pg.return_postgres_cursor()
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
	conn,cursor = pg.return_postgres_cursor()
	engine = pg.return_sql_alchemy_engine()

	query = "select min(ver) as ver from annotation.labelled_treatments t1 where label=0 or label=1"
	new_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])+1
	old_version = new_version-1
	conditions_set = get_all_conditions_set()
	treatments_set = get_all_treatments_set()


	query = "select count(*) as cnt from annotation.labelled_treatments where label=0 or label=1"
	max_counter = int(pg.return_df_from_query(cursor, query, None, ['cnt'])['cnt'][0])

	query = """
			select t1.id, condition_id, treatment_id, label, ver from annotation.labelled_treatments t1 where ver=%s and (label=0 or label=1) limit 1
		"""
	labels_df = pg.return_df_from_query(cursor, query, (old_version,), ['id', 'condition_id', 'treatment_id', 'label', 'ver'])
	
	while (len(labels_df) > 0):
		print(labels_df)
		write_sentence_vectors_from_labels(labels_df, conditions_set, treatments_set, write_type, engine, cursor)
		query = "UPDATE annotation.labelled_treatments set ver = %s where id = %s"
		cursor.execute(query, (new_version, labels_df['id'].values[0]))
		cursor.connection.commit()

		query = """
			select t1.id, condition_id, treatment_id, label, ver from annotation.labelled_treatments t1 where ver=%s and (label=0 or label=1) limit 1
		"""
		labels_df = pg.return_df_from_query(cursor, query, (old_version,), ['id', 'condition_id', 'treatment_id', 'label', 'ver'])

def gen_datasets_bottom(new_version, conditions_set, treatments_set):
	number_of_processes = 1
	engine_arr = []
	cursor_arr = []
	old_version = new_version-1

	for i in range(number_of_processes):
		engine = pg.return_sql_alchemy_engine()
		engine_arr.append(engine)
		conn,cursor = pg.return_postgres_cursor()
		cursor_arr.append(cursor)

	task_queue = mp.Queue()

	pool = []
	for i in range(number_of_processes):
		p = mp.Process(target=gen_dataset_worker, args=(task_queue, engine_arr[i], cursor_arr[i]))
		pool.append(p)
		p.start()

	counter = 0

	conn,cursor = pg.return_postgres_cursor()

	query = """
			select t1.id, condition_id, treatment_id, label, ver from annotation.labelled_treatments t1 where ver=%s and condition_id=%s limit 2
		"""
	labels_df = pg.return_df_from_query(cursor, query, (old_version,'%'), ['id', 'condition_id', 'treatment_id', 'label', 'ver'])
	

	while (len(labels_df) > 0 and counter <= 1):
		query = "UPDATE annotation.labelled_treatments set ver = %s where id = %s"
		cursor.execute(query, (new_version, labels_df['id'].values[0]))
		cursor.connection.commit()

		params = (labels_df, conditions_set, treatments_set)
		task_queue.put((write_sentence_vectors_from_labels, params))

		print(labels_df)

		query = """
			select t1.id, condition_id, treatment_id, label, ver from annotation.labelled_treatments t1 where ver=%s and condition_id=%s limit 2
		"""

		labels_df = pg.return_df_from_query(cursor, query, (old_version,'%'), ['id', 'condition_id', 'treatment_id', 'label', 'ver'])
		
		# except:
		# 	cursor.connection.rollback()
		counter += 1
	cursor.close()
	conn.close()

	for i in range(number_of_processes):
		task_queue.put('STOP')

	for p in pool:
		p.join()

	for g in engine_arr:
		g.dispose()

	for c in cursor_arr:
		c.close()


def gen_dataset_worker(input, engine, cursor):
	for func,args in iter(input.get, 'STOP'):
		gen_dataset_calculate(func, args, engine, cursor)

def gen_dataset_calculate(func, args, engine, cursor):
	func(*args, engine, cursor)

def write_sentence_vectors_from_labels(labels_df, conditions_set, treatments_set, write_type, engine, cursor):
	condition_sentence_ids = pd.DataFrame()

	for index,item in labels_df.iterrows():
		sentences_df = pd.DataFrame()

		if item['condition_id'] == '%':
			tx_id_arr = [item['treatment_id']]
			tx_id_arr.extend(ann.get_children(item['treatment_id'], cursor))
			tx_id_arr = ann.get_concept_synonyms_list_from_list(tx_id_arr, cursor)
	
			sentences_query = """
						select 
							t1.sentence_tuples
							,t1.sentence
							,t2.condition_id
							,t1.conceptid as treatment_id
							,t1.id
							,t1.pmid
						from annotation.sentences5 t1
						right join (select id, conceptid as condition_id from annotation.sentences5) t2
							on t1.id = t2.id
						inner join (select root_cid from annotation.concept_types where rel_type='condition') t3
							on t2.condition_id = t3.root_cid
						where t1.conceptid in %s
					"""
			sentences_df = pg.return_df_from_query(cursor, sentences_query, (tuple(tx_id_arr),), \
				['sentence_tuples', 'sentence', 'condition_id', 'treatment_id', 'id', 'pmid'])
			
			print("finished query")
			
			print("len of wildcard: " + str(len(sentences_df)))
		else:
			condition_id_arr = [item['condition_id']]
			condition_id_arr.extend(ann.get_children(item['condition_id'], cursor))
			condition_id_arr = ann.get_concept_synonyms_list_from_list(condition_id_arr, cursor)

			tx_id_arr = [item['treatment_id']]
			tx_id_arr.extend(ann.get_children(item['treatment_id'], cursor))
			tx_id_arr = ann.get_concept_synonyms_list_from_list(tx_id_arr, cursor)

			sentences_query = """
				select sentence_tuples, sentence, t2.condition_id, t1.conceptid as treatment_id, t1.id, t1.pmid
				from annotation.sentences5 t1
				right join (
					select id, conceptid as condition_id from annotation.sentences5 where conceptid in %s
				) t2
				on t1.id = t2.id
				where conceptid in %s
			"""
			sentences_df = pg.return_df_from_query(cursor, sentences_query, (tuple(condition_id_arr), tuple(tx_id_arr)), \
				['sentence_tuples', 'sentence', 'condition_id', 'treatment_id', 'id', 'pmid'])
		
		
		if write_type == 'gen':
			sentences_df['x_train_gen'] = sentences_df.apply(apply_get_generic_labelled_data, all_conditions_set=conditions_set, all_treatments_set=treatments_set, axis=1)
			sentences_df['x_train_spec'] = sentences_df.apply(apply_get_specific_labelled_data, axis=1)
			sentences_df['label'] = item['label']
			sentences_df['ver_gen'] = 0
			sentences_df['ver_spec'] = 0
			sentences_df = sentences_df[['id', 'sentence', 'sentence_tuples', 'condition_id', 'treatment_id', 'x_train_gen', 'x_train_spec', 'label', 'ver_gen', 'ver_spec']]
			sentences_df.to_sql('training_sentences_with_version_v2', engine, schema='annotation', if_exists='append', \
				index=False, dtype={'sentence_tuples' : sqla.types.JSON, 'x_train_gen' : sqla.types.JSON, 'x_train_spec' : sqla.types.JSON, 'sentence' : sqla.types.Text})
		elif write_type == 'spec':
			for i,d in sentences_df.iterrows():
				res = get_labelled_data_sentence_custom_v2(d, d['condition_id'], d['treatment_id'])
				res['label'] = item['label']
				res['ver'] = 0
				res.to_sql('custom_training_sentences', engine, schema='annotation', if_exists='append', \
						index=False, dtype={'sentence_tuples' : sqla.types.JSON, 'x_train_cust' : sqla.types.JSON, 'sentence' : sqla.types.Text})

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

	all_conditions_set = get_all_conditions_set()
	all_treatments_set = get_all_treatments_set()

	# should be OK to load 10k into memory

	testing_query = "select sentence, condition_id, treatment_id, x_train_gen, x_train_spec, label from annotation.test_sentences_v2 limit 200"
	sentences_df = pg.return_df_from_query(cursor, testing_query, None, \
		['sentence', 'condition_id', 'treatment_id', 'x_train_gen', 'x_train_spec', 'label'])

	total = len(sentences_df)
	zero_zero = 0
	zero_one = 0
	one_zero = 0
	one_one = 0

	for ind,item in sentences_df.iterrows():
		x_train_gen = np.array([item['x_train_gen']])
		# x_train_spec = np.array([item['x_train_spec']])
		# res = float(model.predict([x_train_gen, x_train_spec])[0][0])
		res = float(model.predict([x_train_gen])[0][0])

		if ((item['label'] == 1) and (res >= 0.50)):
			one_one += 1
		elif((item['label'] == 1) and (res < 0.50)):
			one_zero += 1
			print("label one, model zero")
			print(res)
			print(item['sentence'])
			print(item['condition_id'])
			print(item['treatment_id'])
			print("======================")
		elif ((item['label'] == 0) and (res < 0.50)):
			zero_zero += 1
		elif ((item['label'] == 0) and (res >= 0.50)):
			zero_one += 1
			print("label zero, model one")
			print(res)
			print(item['sentence'])
			print(item['condition_id'])
			print(item['treatment_id'])
			print("======================")
		
	u.pprint("label 1, res 1: " + str(one_one))
	u.pprint("label 1, res 0: " + str(one_zero))
	u.pprint("label 0, res 0: " + str(zero_zero))
	u.pprint("label 0, res 1: " + str(zero_one))

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
	# conn, cursor = pg.return_postgres_cursor()
	# print(get_word_index('195967001', cursor))
	# print(train_data_generator(10, cursor))
	# train_with_word2vec()
	# parallel_treatment_recategorization_top('double-01.hdf5')
	# gen_datasets_top('gen')
	# update_word2vec('model.3.hdf5')


	train_with_word2vec_v2()
	

	# load_word_counts()
	# input_sentence = input("Enter sentence: ")
	# term = ann.clean_text(input_sentence)
	# all_words = ann.get_all_words_list(term)
	# cache = ann.get_cache(all_words, False, cursor)
	
	# annotation, sentences = ann.annotate_text_not_parallel(input_sentence, 'unlabelled', cache, cursor, True, True, False)
	# annotation = ann.acronym_check(annotation)
	# sentence_tuple = ann.get_sentence_annotation(term, annotation)
	# u.pprint(sentence_tuple)
	# print(len(sentence_tuple))

	# condition_ind = int(input("Enter condition index (start or end): "))
	# print(sentence_tuple[condition_ind])

	# condition_id = sentence_tuple[condition_ind][1]

	# print(condition_id)
	
	# sentence_df = pd.DataFrame([[term, sentence_tuple]], columns=['sentence', 'sentence_tuples'])
	# print(sentence_df)

	# final_res = analyze_sentence(sentence_df, condition_id, cursor)
	
	
	# cursor.close()
	# conn.close()

	# print(final_res)

	# print_contingency('m01.16.20-08.hdf5')
	# print_contingency('double-10.hdf5')
	# print_contingency('double-02.hdf5')			

