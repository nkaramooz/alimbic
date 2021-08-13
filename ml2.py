import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from nltk.stem.wordnet import WordNetLemmatizer
from operator import itemgetter
import nltk.data
import numpy as np
import time
import utilities.pglib as pg
import utilities.utils2 as u2
import collections
import os
import random
import sys
import snomed_annotator2 as ann2
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
import datetime
import tensorflow as tf
from gensim.models import Word2Vec


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
	query = "select root_acid from annotation.concept_types where (rel_type='condition' or rel_type='symptom') and active=1"
	conn,cursor = pg.return_postgres_cursor()
	all_conditions_set = set(pg.return_df_from_query(cursor, query, None, ['root_acid'])['root_acid'].tolist())
	return all_conditions_set


def get_all_treatments_set():
	query = "select root_acid from annotation.concept_types where rel_type='treatment' and active=1"
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
		curr_version = int(pg.return_df_from_query(cursor, "select min(ver) from ml2.train_sentences", \
			None, ['ver'])['ver'][0])

		new_version = curr_version + 1
		
		query = "select id, x_train_gen, x_train_spec, x_train_gen_mask, x_train_spec_mask, label from ml2.train_sentences where ver = %s limit %s"
		train_df = pg.return_df_from_query(cursor, query, (curr_version, batch_size), ['id', 'x_train_gen', 'x_train_spec', 'x_train_gen_mask', 'x_train_spec_mask', 'label'])

		x_train_gen = train_df['x_train_gen'].tolist()
		x_train_spec = train_df['x_train_spec'].tolist()
		x_train_mask = train_df['x_train_gen_mask'].tolist()


		y_train = train_df['label'].tolist()
		

		try:
			id_df = train_df['id'].tolist()
			query = """
				UPDATE ml2.train_sentences
				SET ver = %s
				where id = ANY(%s);
			"""
			cursor.execute(query, (new_version, id_df))
			cursor.connection.commit()


			yield (np.asarray(x_train_gen), np.asarray(y_train))
		except:
			print("update version failed. Rolling back")
			cursor.connection.rollback()
			yield None

	
def train_with_rnn(max_cnt):
	conn,cursor = pg.return_postgres_cursor()

	embedding_size=300
	batch_size = 500
	num_epochs = 20

	model_input_gen = Input(shape=(max_words,))
	model_gen_emb = Embedding(vocabulary_size+1, embedding_size, trainable=True, mask_zero=True)(model_input_gen)
	lstm_model = LSTM(500, recurrent_dropout=0.3, return_sequences=True)(model_gen_emb)
	lstm_model = LSTM(500, recurrent_dropout=0.3, return_sequences=True)(lstm_model)
	lstm_model = LSTM(300, recurrent_dropout=0.3)(lstm_model)
	lstm_model = Dropout(0.3)(lstm_model)
	lstm_model = Dense(100)(lstm_model)
	pred = Dense(1, activation='sigmoid')(lstm_model)
	
	model = Model(inputs=[model_input_gen], outputs=[pred])

	print(model.summary())
	# model.summary(print_fn=lambda x: report.write(x + '\n'))
	# report.write('\n')

	model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

	checkpointer = ModelCheckpoint(filepath='./gen_500_{epoch:02d}.hdf5', verbose=1)

	log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

	history = model.fit(train_data_generator_v2(batch_size, cursor), \
	 epochs=num_epochs, class_weight={0:1, 1:70}, steps_per_epoch =((max_cnt//batch_size)+1),
	  callbacks=[checkpointer, tensorboard_callback])


def update_rnn(model_name, max_cnt):
	conn,cursor = pg.return_postgres_cursor()
	embedding_size=300
	batch_size = 300
	num_epochs = 10
	model = load_model(model_name)

	checkpointer = ModelCheckpoint(filepath='./emb_500_update_{epoch:02d}.hdf5', verbose=1)

	
	history = model.fit(train_data_generator_v2(batch_size, cursor), \
	 epochs=num_epochs, class_weight={0:1, 1:10}, steps_per_epoch =((max_cnt//batch_size)+1),
	  callbacks=[checkpointer])


def gen_treatment_predictions_top(model_name):
	conn,cursor = pg.return_postgres_cursor()
	query = "select min(ver) from ml2.treatment_dataset_subset_staging"
	new_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])+1
	old_version = new_version-1
	curr_version = old_version

	while curr_version != new_version:
		gen_treatment_predictions_bottom(model_name, old_version, conn, cursor)
		curr_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])
	conn.close()
	cursor.close()


def gen_treatment_predictions_bottom(model_name, old_version, conn, cursor):
	number_of_processes = 4

	task_queue = mp.Queue()

	pool = []
	for i in range(number_of_processes):
		p = mp.Process(target=recat_worker, args=(task_queue,))
		pool.append(p)
		p.start()

	counter = 0

	query = """
		select 
			entry_id
			,x_train_gen
			,condition_acid
			,treatment_acid
			,sentence_tuples
		from ml2.treatment_dataset_subset_staging
		where ver = %s limit 1000
	"""

	treatment_candidates_df = pg.return_df_from_query(cursor, query, \
				(old_version,), ['entry_id','x_train_gen', 'condition_acid', 'treatment_acid', 'sentence_tuples'])

	while (counter < number_of_processes) and (len(treatment_candidates_df.index) > 0):
		params = (model_name, treatment_candidates_df)
		task_queue.put((batch_treatment_predictions, params))

		update_query = """
			set schema 'ml2';
			UPDATE ml2.treatment_dataset_subset_staging
			SET ver = %s
			where entry_id = ANY(%s);
		"""

		entry_id_list = treatment_candidates_df['entry_id'].tolist()
		new_version = old_version + 1
		cursor.execute(update_query, (new_version, entry_id_list))
		cursor.connection.commit()
		treatment_candidates_df = pg.return_df_from_query(cursor, query, \
			(old_version,), ['entry_id', 'x_train_gen', 'condition_acid', 'treatment_acid', 'sentence_tuples'])

		counter += 1

	for i in range(number_of_processes):
		task_queue.put('STOP')

	for p in pool:
		p.join()
	

def batch_treatment_predictions(model_name, treatment_candidates_df):
	model = load_model(model_name)
	conn,cursor = pg.return_postgres_cursor()
	engine = pg.return_sql_alchemy_engine()

	treatment_candidates_df['score'] = treatment_candidates_df.apply(apply_score, model=model, axis=1)
	treatment_candidates_df.to_sql('treatment_recs_staging', engine, schema='ml2', if_exists='append', \
	 index=False, dtype={'sentence_tuples' : sqla.types.JSON})

	cursor.close()
	conn.close()
	engine.dispose()


def parallel_treatment_recategorization_bottom(model_name, old_version, all_conditions_set, all_treatments_set):
	conn,cursor = pg.return_postgres_cursor()
	number_of_processes = 40

	task_queue = mp.Queue()

	pool = []
	for i in range(number_of_processes):
		p = mp.Process(target=recat_worker, args=(task_queue,))
		pool.append(p)
		p.start()

	counter = 0

	query = """
		select 
			entry_id
			,condition_acid
			,treatment_acid
			,sentence_tuples
		from ml2.treatment_candidates_with_entry_id

		where ver = %s limit 5000"""

	treatment_candidates_df = pg.return_df_from_query(cursor, query, \
				(old_version,), ['entry_id', 'condition_acid', 'treatment_acid', 'sentence_tuples'])

	while (counter < number_of_processes) and (len(treatment_candidates_df.index) > 0):
		params = (model_name, treatment_candidates_df, all_conditions_set, all_treatments_set)
		task_queue.put((batch_treatment_recategorization, params))

		update_query = """
			set schema 'ml2';
			UPDATE treatment_candidates_with_entry_id
			SET ver = %s
			where entry_id = ANY(%s);
		"""

		# update_query = """
		# 	set schema 'ml2';
		# 	UPDATE treatment_candidates
		# 	SET ver = %s
		# 	where sentence_id = ANY(%s) and treatment_acid = ANY(%s);
		# """

		entry_id_list = treatment_candidates_df['entry_id'].tolist()
		new_version = old_version + 1
		cursor.execute(update_query, (new_version, entry_id_list))

		cursor.connection.commit()
		treatment_candidates_df = pg.return_df_from_query(cursor, query, \
			(old_version,), ['entry_id', 'condition_acid', 'treatment_acid', 'sentence_tuples'])

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


def apply_score(row, model):
	gen = np.array([row['x_train_gen']])
	# spec = np.array([row['x_train_spec']])
	# mask = np.array([row['x_train_mask']])


	res = float(model.predict([gen])[0][0])
	return res


def batch_treatment_recategorization(model_name, treatment_candidates_df, all_conditions_set, all_treatments_set):
	model = load_model(model_name)
	conn,cursor = pg.return_postgres_cursor()
	engine = pg.return_sql_alchemy_engine()

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
mask_generic_condition_marker = 3
mask_generic_treatment_marker = 4
# sample[0] = dict value of item
# sample[1]: 1=target_condition, 2=target_treatment
def get_labelled_data_sentence_generic_v2(sentence, conditions_set, all_treatments_set):
	conn,cursor = pg.return_postgres_cursor()
	condition_id = sentence['condition_acid']
	tx_id = sentence['treatment_acid']

	final_results = pd.DataFrame()
	sample = [0]*max_words
	sample_spec = [0]*max_words
	generic_mask = [0]*max_words
	spec_mask = [0]*max_words
	counter = 0

	for index,words in enumerate(sentence['sentence_tuples']):	

		if (conditions_set is not None) and words[1] == condition_id and sample[counter-1] == target_condition_key:
			continue
		elif (conditions_set is not None) and words[1] == condition_id and sample[counter-1] != target_condition_key:
			sample[counter] = target_condition_key
			sample_spec[counter] = get_word_index(words[1], cursor)
			generic_mask[counter] = mask_condition_marker
			spec_mask[counter] = mask_condition_marker
		elif (conditions_set is not None) and (words[1] in conditions_set) and (sample[counter-1] == generic_condition_key):
			continue
		elif (conditions_set is not None) and (words[1] in conditions_set) and (sample[counter-1] != generic_condition_key):
			sample[counter] = generic_condition_key
			sample_spec[counter] = get_word_index(words[1], cursor)
			generic_mask[counter] = mask_generic_condition_marker
		elif (tx_id is not None) and (words[1] == tx_id and (sample[counter-1] == target_treatment_key)):
			continue
		elif (tx_id is not None) and (words[1] == tx_id) and (sample[counter-1] != target_treatment_key):
			sample[counter] = target_treatment_key
			sample_spec[counter] = get_word_index(words[1], cursor)
			generic_mask[counter] = mask_treatment_marker
			spec_mask[counter] = mask_treatment_marker
		elif (all_treatments_set is not None) and (words[1] in all_treatments_set) and (sample[counter-1] == generic_treatment_key):
			continue
		elif (all_treatments_set is not None) and (words[1] in all_treatments_set) and (sample[counter-1] != generic_treatment_key):
			sample[counter] = generic_treatment_key
			sample_spec[counter] = get_word_index(words[1], cursor)
			generic_mask[counter] = mask_generic_treatment_marker
		elif (words[1] != 0) and (get_word_index(words[1], cursor) != UNK_ind) and (sample[counter-1] != get_word_index(words[1], cursor)):
			sample[counter] = get_word_index(words[1], cursor)
			sample_spec[counter] = sample[counter]
		elif (words[1] == 0) and (words[0] == 'caused' or words[0] == 'causing' or words[0] == 'causes' or words[0] == 'induce'):
			sample[counter] = get_word_index('cause', cursor)
			sample_spec[counter] = sample[counter]
		elif (words[1] == 0) and (get_word_index(words[0], cursor) != UNK_ind) and (sample[counter-1] != get_word_index(words[0], cursor)):
			sample[counter] = get_word_index(words[0], cursor)
			sample_spec[counter] = sample[counter]
		elif ((words[1] != 0 and (get_word_index(words[1], cursor) == UNK_ind)) or (words[1] == 0 and get_word_index(words[0], cursor) == UNK_ind)):
			sample[counter] = UNK_ind
			sample_spec[counter] = UNK_ind
		else:
			counter -= 1

		counter += 1

		if counter >= max_words-1:
			break

	cursor.close()
	conn.close()
	sentence['x_train_gen'] = sample
	sentence['x_train_gen_mask'] = generic_mask
	sentence['x_train_spec_mask'] = spec_mask
	sentence['x_train_spec'] = sample_spec

	return sentence

			


# used for analyzing sentence
def get_labelled_data_sentence_generic_v2_custom(sentence, condition_id, tx_id, conditions_set, all_treatments_set):
	conn,cursor = pg.return_postgres_cursor()
	final_results = pd.DataFrame()
	sample_gen = [0]*max_words
	sample_spec = [0]*max_words
	generic_mask = [0]*max_words
	counter = 0

	for index,words in enumerate(sentence['sentence_tuples'][0]):
		if words[1] == condition_id and sample_gen[counter-1] == target_condition_key:
			continue
		elif words[1] == condition_id and sample_gen[counter-1] != target_condition_key:
			sample_gen[counter] = target_condition_key
			sample_spec[counter] = get_word_index(words[1], cursor)
			generic_mask[counter] = mask_condition_marker
		elif (words[1] in conditions_set) and (sample_gen[counter-1] == generic_condition_key):
			continue
		elif (words[1] in conditions_set) and (sample_gen[counter-1] != generic_condition_key):
			sample_gen[counter] = generic_condition_key
			sample_spec[counter] = get_word_index(words[1], cursor)
			generic_mask[counter] = mask_generic_condition_marker
		elif (words[1] == tx_id) and (sample_gen[counter-1] != target_treatment_key):
			sample_gen[counter] = target_treatment_key
			sample_spec[counter] = get_word_index(words[1], cursor)
			generic_mask[counter] = mask_treatment_marker
		elif (words[1] in all_treatments_set) and (sample_gen[counter-1] == generic_treatment_key):
			continue
		elif (words[1] in all_treatments_set) and (sample_gen[counter-1] != generic_treatment_key):
			sample_gen[counter] = generic_treatment_key
			sample_spec[counter] = get_word_index(words[1], cursor)
			# generic_mask[counter] = mask_generic_treatment_marker
		elif (words[1] != 0) and (get_word_index(words[1], cursor) != UNK_ind) and (sample_gen[counter-1] != get_word_index(words[1], cursor)):
			sample_gen[counter] = get_word_index(words[1], cursor)
			sample_spec[counter] = get_word_index(words[1], cursor)
		elif (words[1] == 0) and (get_word_index(words[0], cursor) != UNK_ind) and (sample_gen[counter-1] != get_word_index(words[0], cursor)):
			sample_gen[counter] = get_word_index(words[0], cursor)
			sample_spec[counter] = get_word_index(words[0], cursor)
		elif ((words[1] == 0) and (get_word_index(words[0], cursor) == UNK_ind)) or ((words[1] == 1) \
			and (get_word_index(words[1], cursor) == UNK_ind)):
			sample_gen[counter] = UNK_ind
			sample_spec[counter] = UNK_ind
		else:
			counter -= 1

		counter += 1

		if counter >= max_words-1:
			break

	cursor.close()
	conn.close()
	return sample_gen, sample_spec, generic_mask

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

	get_query = """
			select t1.id, condition_acid::text, treatment_acid::text, label, ver 
			from ml2.labelled_treatments t1 where ver=%s and (label=0 or label=1) 
			and treatment_acid in (select root_acid from annotation.concept_types where rel_type='treatment') limit %s
		"""
	labels_df = pg.return_df_from_query(cursor, get_query, (old_version, number_of_processes), ['id', 'condition_acid', 'treatment_acid', 'label', 'ver'])
	

	while len(labels_df.index) > 0:
		for index,item in labels_df.iterrows():
			params = (item['condition_acid'], item['treatment_acid'], item['label'])
			task_queue.put((gen_datasets_mp_bottom, params))

		update_query = "UPDATE ml2.labelled_treatments set ver = %s where id in %s"
		cursor.execute(update_query, (new_version, tuple(labels_df['id'].values.tolist())))
		cursor.connection.commit()

		labels_df = pg.return_df_from_query(cursor, get_query, (old_version, number_of_processes), ['id', 'condition_acid', 'treatment_acid', 'label', 'ver'])
		
		
	for i in range(number_of_processes):
		task_queue.put('STOP')

	for p in pool:
		p.join()

	for p in pool:
		p.close()

	cursor.close()
	conn.close()


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
						from pubmed.sentence_concept_arr_1_9 t1
						join (select root_acid as condition_acid from annotation.concept_types where rel_type='condition' or rel_type='symptom' or rel_type='cause') t2
							on t2.condition_acid = ANY(t1.concept_arr::text[])
						join (select acid as treatment_acid from annotation.downstream_root_cid where acid in %s) t3
							on t3.treatment_acid = ANY(t1.concept_arr::text[])
						join pubmed.sentence_tuples_1_9 t4
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
				from pubmed.sentence_concept_arr_1_9 t1
				join (select acid as condition_acid from annotation.downstream_root_cid where acid in %s) t2
					on t2.condition_acid = ANY(t1.concept_arr::text[])
				join (select acid as treatment_acid from annotation.downstream_root_cid where acid in %s) t3
					on t3.treatment_acid = ANY(t1.concept_arr::text[])
				join pubmed.sentence_tuples_1_9 t4
					on t1.sentence_id = t4.sentence_id
			"""
		sentences_df = pg.return_df_from_query(cursor, sentences_query, (tuple(condition_id_arr), tuple(tx_id_arr)), \
				['sentence_tuples', 'condition_acid', 'treatment_acid', 'sentence_id', 'section_ind', 'pmid'])


	cursor.close()
	conn.close()

	if len(sentences_df.index) > 0:
		sentences_df['label'] = label
		write_sentence_vectors_from_labels(sentences_df, conditions_set, treatments_set, 'gen')


def gen_treatment_data_top():
	conn,cursor = pg.return_postgres_cursor()
	
	all_conditions_set = get_all_conditions_set() 
	all_treatments_set = get_all_treatments_set()


	query = "select min(ver) from ml2.treatment_candidates_1_9"
	new_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])+1
	old_version = new_version-1
	curr_version = old_version

	while curr_version != new_version:
		gen_treatment_data_bottom(old_version, new_version, all_conditions_set, all_treatments_set)
		curr_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])


		
def gen_treatment_data_bottom(old_version, new_version, all_conditions_set, all_treatments_set):
	number_of_processes = 35
	conditions_set = get_all_conditions_set()
	treatments_set = get_all_treatments_set()

	task_queue = mp.Queue()
	pool = []

	for i in range(number_of_processes):
		p = mp.Process(target=gen_dataset_worker, args=(task_queue, conditions_set, treatments_set))
		pool.append(p)
		p.start()

	conn,cursor = pg.return_postgres_cursor()

	get_query = """
		select 
			entry_id
			,condition_acid
			,treatment_acid
			,sentence_tuples
		from ml2.treatment_candidates_1_9

		where ver = %s limit 1000"""

	treatment_candidates_df = pg.return_df_from_query(cursor, get_query, \
				(old_version,), ['entry_id', 'condition_acid', 'treatment_acid', 'sentence_tuples'])

	counter = 0

	while (counter < number_of_processes) and (len(treatment_candidates_df.index) > 0):
		params = (treatment_candidates_df,)
		task_queue.put((gen_treatment_dataset, params))
		update_query = """
			set schema 'ml2';
			UPDATE treatment_candidates_1_9
			SET ver = %s
			where entry_id = ANY(%s);
		"""
		cursor.execute(update_query, (new_version, treatment_candidates_df['entry_id'].values.tolist(), ))
		cursor.connection.commit()

		treatment_candidates_df = pg.return_df_from_query(cursor, get_query, \
				(old_version,), ['entry_id', 'condition_acid', 'treatment_acid', 'sentence_tuples'])

		counter += 1

	for i in range(number_of_processes):
		task_queue.put('STOP')

	for p in pool:
		p.join()

	for p in pool:		p.close()

	cursor.close()
	conn.close()

def gen_treatment_dataset(treatment_candidates_df, all_conditions_set, all_treatments_set):
	conn,cursor = pg.return_postgres_cursor()
	engine = pg.return_sql_alchemy_engine()

	treatment_candidates_df = treatment_candidates_df.apply(apply_get_generic_labelled_data, \
	 all_conditions_set=all_conditions_set, all_treatments_set=all_treatments_set, axis=1)
	treatment_candidates_df['ver'] = 0
	treatment_candidates_df.to_sql('treatment_dataset_staging', engine, schema='ml2', if_exists='append', \
	 index=False, dtype={'sentence_tuples' : sqla.types.JSON, 'x_train_gen' : sqla.types.JSON})

	cursor.close()
	conn.close()
	engine.dispose()


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

			sentences_df['ver'] = 0
			sentences_df = sentences_df[['sentence_id', 'sentence_tuples', 'condition_acid', 'treatment_acid',\
			 'x_train_gen', 'x_train_gen_mask', 'x_train_spec_mask','x_train_spec', 'label', 'ver']]
			sentences_df.to_sql('training_sentences_staging', engine, schema='ml2', if_exists='append', index=False, \
				dtype={'sentence_tuples' : sqla.types.JSON, 'x_train_gen' : sqla.types.JSON, 'x_train_gen_mask' : sqla.types.JSON, \
					'x_train_spec_mask' : sqla.types.JSON, 'x_train_spec' : sqla.types.JSON, 'sentence' : sqla.types.Text})
			
		cursor.close()
		conn.close()
		engine.dispose()


def analyze_sentence(model_name, sentence, condition_id):
	term = ann2.clean_text(sentence)
	all_words = ann2.get_all_words_list(term)
	cache = ann2.get_cache(all_words, False)
	item = pd.DataFrame([[term, 'title', 0, 0]], columns=['line', 'section', 'section_ind', 'ln_num'])
	sentence_annotations_df, sentence_tuples_df, sentence_concept_arr_df = ann2.annotate_text_not_parallel(item, cache, True, True, True)
	
	model = load_model(model_name)

	all_conditions_set = get_all_conditions_set()
	all_treatments_set = get_all_treatments_set()

	final_res = []

	for ind,word in enumerate(sentence_tuples_df['sentence_tuples'][0]):
		if word[1] == condition_id:
			continue
		elif word[1] != 0:

			sample_gen,sample_spec, mask = get_labelled_data_sentence_generic_v2_custom(sentence_tuples_df, condition_id, word[1], \
				all_conditions_set, all_treatments_set)
			# t1= [49996,1709,1271,1289,12,500,6,49999,9,49998,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
			sample_gen_arr = np.array([sample_gen])
			print(sample_gen_arr)
			# sample_spec_arr = np.array([sample_spec])
			# mask_arr = np.array([mask])

			# res = float(model.predict([sample_gen_arr, sample_spec_arr, mask_arr]))
			res = float(model.predict([sample_gen_arr]))
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

	testing_query = "select id, sentence_id, sentence_tuples,condition_acid, treatment_acid, x_train_gen, x_train_gen_mask, label from ml2.test_sentences_subset where ver=%s"
	sentences_df = pg.return_df_from_query(cursor, testing_query, (curr_version,), \
		['id', 'sentence_id','sentence_tuples','condition_acid', 'treatment_acid', 'x_train_gen','x_train_gen_mask', 'label'])

	zero_zero = 0
	zero_one = 0
	one_zero = 0
	one_one = 0

	for ind,item in sentences_df.iterrows():
		x_train_gen = np.array([item['x_train_gen']])
		x_train_mask = np.array([item['x_train_gen_mask']])

		res = float(model.predict([x_train_gen, x_train_mask])[0][0])

		if ((item['label'] == 1) and (res >= 0.50)):
			one_one += 1
		elif((item['label'] == 1) and (res < 0.50)):
			print("below label=1 and res < 0.50")
			u2.pprint(item['condition_acid'])
			u2.pprint(item['treatment_acid'])
			u2.pprint(item['sentence_tuples'])
			print("end")
			one_zero += 1
		elif ((item['label'] == 0) and (res < 0.50)):
			zero_zero += 1
		elif ((item['label'] == 0) and (res >= 0.50)):
			print("below label=0, res >= 0.50")
			u2.pprint(item['condition_acid'])
			u2.pprint(item['treatment_acid'])
			u2.pprint(item['sentence_tuples'])
			print("end")
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



if __name__ == "__main__":
	# caused by and associated with not working well
	# sentence = "Montelukast as a treatment for acute interstitial nephritis"
	# sentence = "metoprolol improves cough in severe acute interstitial nephritis caused by amiodarone"
	# sentence = "Rare allergic reaction of the kidney: sitagliptin-induced acute tubulointerstitial nephritis"
	# sentence = "Acute Interstitial Nephritis Associated with Sofosbuvir and Daclatasvir"
	# sentence = "Acute Interstitial Nephritis caused by Sofosbuvir and Daclatasvir"
	# sentence = "Acute Interstitial Nephritis associated with Sofosbuvir"
	# sentence = "Sofosbuvir and Daclatasvir causes Acute Interstitial Nephritis"
	# sentence = "Amiodarone induced acute interstitial nephritis"
	# sentence = "acute interstitial nephritis caused by amiodarone"
	# sentence = "amiodarone for acute interstitial nephritis"
	
	# sentence = "acute interstitial nephritis associated with aerosolized pentamidine"
	# sentence = "hepatic and splenic blush on computed tomography in children following blunt force acute interstitial nephritis"
	# sentence = "Acute interstitial nephritis induced by midazolam and abolished by flumazenil"
	# sentence = "Acute interstitial nephritis resolved after with ipecac"
	# sentence = "Effect of dexamethasone on complication rate and mortality in patients with acute interstitial nephritis"
	# sentence = "Enteral nutrition tube placement assisted by ultrasonography in patients with acute interstitial nephritis"
	# sentence = "acute interstitial nephritis following catheter ablation for atrial fibrillation"
	# sentence = "Successful treatment of acute interstitial nephritis by cervical esophageal ligation and decompression"
	# sentence = "We report a case of relapse of mucosal acute interstitial nephritis after aggresive immunotherapy for ankylosing spondylitis with requirement for secondary prophylaxis with amphotericin B to prevent reactivation"
	# sentence = "New onset acute interstitial nephritis associated with use of soy isoflavone supplements"
	# sentence = sentence.lower()
	# model_name = 'emb_500_update_04.hdf5'
	# model_name = 'gen_500_20.hdf5'

	# 8 can get the associated with concept
	# condition_id = '10609'
	# analyze_sentence(model_name, sentence, condition_id)

	# word2vec_emb_top()
	# build_w2v_embedding()

	

	# sent_query = "select sentence_id, x_train_spec from ml2.w2v_emb_train_w_ver where ver=%s limit 1"
	# sentences_df = pg.return_df_from_query(cursor, sent_query, (0,), ['sentence_id', 'x_train_spec'])
	# sentences_list = sentences_df['x_train_spec'].tolist()
	# sentence_id_list = sentences_df['sentence_id'].tolist()
		
		# # Need to convert 
	# for c1,i in enumerate(sentences_list):
	# 	for c2,j in enumerate(i):
	# 		sentences_list[c1][c2]=str(j)
	# print(sentences_list)

	# model = Word2Vec(vector_size=500, window=5, min_count=1, negative=15, epochs=5, workers=20)
	# model.build_vocab(sentences_list)
	# print(model)
	# parallel_treatment_recategorization_top('../double-19.hdf5')

	# parallel_treatment_recategorization_top('emb_500_update_04.hdf5')

	# gen_treatment_data_top()
	# gen_treatment_predictions_top('emb_500_update_04.hdf5')
	# gen_datasets_mp(1)

	
	conn, cursor = pg.return_postgres_cursor()

	max_cnt = int(pg.return_df_from_query(cursor, "select count(*) as cnt from ml2.train_sentences", \
			None, ['cnt'])['cnt'][0])
	cursor.close()
	conn.close()
	# update_rnn('gen_500_20.hdf5', max_cnt)
	train_with_rnn(max_cnt)
	# print_contingency('gen_500_24.hdf5')