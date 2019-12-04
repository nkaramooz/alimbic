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
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
import sqlalchemy as sqla
import multiprocessing as mp


# index 0 = filler
# index 1 = UNK

spacer_ind = 0
UNK_ind = 1
vocabulary_size = 20000

# need one for root (condition), need one for rel treatment
# need one for none (spacer)
vocabulary_spacer = 5
target_treatment_key = vocabulary_size-vocabulary_spacer+3
target_condition_key = vocabulary_size-vocabulary_spacer+2
generic_condition_key = vocabulary_size-vocabulary_spacer+1
generic_treatment_key = vocabulary_size-vocabulary_spacer

max_words = 60

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

	while len(sentence_df) > 0:
		sentence_array = sentence_df['sentence_tuples'].tolist()
		apply_word_counts_to_dict(cursor, sentence_array, all_conditions_set)
		update_query = "UPDATE annotation.distinct_sentences set version = %s where id = %s"
		cursor.execute(update_query, (new_version, sentence_df['id'].values[0]))
		cursor.connection.commit()
		sentence_df = pg.return_df_from_query(cursor, query, None, ['id', 'sentence_tuples', 'version'])


# {word : count} in current batch
## need to update postgres table with 

# tmp counts
# select union sum word
# drop tmp count

def apply_word_counts_to_dict(cursor, sentence_row, all_conditions_set):
	last_word = None
	for index,words in enumerate(sentence_row[0]):
		word_key = None
		if words[1] != 0 and last_word != words[1]:
			if words[1] in all_conditions_set:
				# update_counts(counts, generic_condition_key)
				word_key = generic_condition_key
			else:
				# update_counts(counts, words[1])
				word_key = words[1]
			last_word = words[1]
		elif words[1] == 0 and words[0] != last_word:
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


def get_word_index(word):
	conn,cursor = pg.return_postgres_cursor()
	query = "select rn from annotation.word_counts_20k where word=%s limit 1"
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
		query = "select word from annotation.word_counts_20k where rn = %s limit 1"
		word_df = pg.return_df_from_query(cursor, query, (index,), ['word'])
		return str(word_df['word'][0])


def train_data_generator(batch_size, cursor):

	while True:
		curr_version = int(pg.return_df_from_query(cursor, "select min(ver) from annotation.training_sentences_with_version", \
			None, ['ver'])['ver'][0])
		batch_size_0 = int(batch_size*0.80)
		batch_size_1 = batch_size - batch_size_0

		query = "select id, x_train, label from annotation.training_sentences_with_version where ver=%s and label=0 order by random() limit %s"
		train_df_0 = pg.return_df_from_query(cursor, query, (curr_version, batch_size_0), ['id', 'x_train', 'label'])

		x_train = train_df_0['x_train'].tolist()
		y_train = train_df_0['label'].tolist()
		id_df = train_df_0['id'].tolist()

		query = "select id, x_train, label from annotation.training_sentences_with_version where label = 1 order by random() limit %s"
		train_df_1 = pg.return_df_from_query(cursor, query, (batch_size_1,), ['id', 'x_train', 'label'])

		x_train.extend(train_df_1['x_train'].tolist())
		y_train.extend(train_df_1['label'].tolist())
		id_df.extend(train_df_1['id'].tolist())

		try:
			new_version = curr_version + 1
			query = """
				set schema 'annotation';
				UPDATE annotation.training_sentences_with_version
				SET ver = %s
				where id in %s
			"""
			cursor.execute(query, (new_version, tuple(id_df)))
			cursor.connection.commit()
			yield (np.asarray(x_train), np.asarray(y_train))
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
	num_epochs = 2

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

	print(model.summary())
	model.summary(print_fn=lambda x: report.write(x + '\n'))
	report.write('\n')

	model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])
	checkpointer = ModelCheckpoint(filepath='./model-{epoch:02d}.hdf5', verbose=1)
	# class_weight={0 : 0.77, 1 : 1}
	history = model.fit_generator(train_data_generator(batch_size, cursor), \
	 epochs=num_epochs, class_weight={0:1, 1:1}, steps_per_epoch =((2810596//batch_size)+1),callbacks=[checkpointer])


	loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
	training = "Training Accuracy: {:.4f}".format(accuracy)
	print(training)
	loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
	testing = "Testing Accuracy:  {:.4f}".format(accuracy)
	print(testing)
	# plot_history(history)

	model.save('txp_200_07_27_w2v.h5')
	report.write(training)
	report.write('\n')
	report.write(testing)
	report.write('\n')
	report.close()
	

def parallel_treatment_recategorization_top(model_name):
	conn,cursor = pg.return_postgres_cursor()

	# dictionary, reverse_dictionary = get_dictionaries()
	conditions_query = "select root_cid from annotation.concept_types where rel_type='condition' or rel_type='symptom' "
	all_conditions_set = set(pg.return_df_from_query(cursor, conditions_query, None, ["root_cid"])["root_cid"].tolist())

	all_treatments_set = get_all_treatments_set()

	max_query = "select count(*) as cnt from annotation.title_treatment_candidates_filtered_final where condition_id = '91302008' "
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
			where condition_id = '91302008' limit 2000 offset %s"""
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


def apply_get_labelled_data(row, all_conditions_set, all_treatments_set):
	return get_labelled_data_sentence(row, row['condition_id'], row['treatment_id'], all_conditions_set, all_treatments_set)


def apply_score(row, model):
	res = float(model.predict(np.array([row['x_train']]))[0][0])
	return res


def batch_treatment_recategorization(model_name, treatment_candidates_df, all_conditions_set, all_treatments_set, engine):
	model = load_model(model_name)
	treatment_candidates_df['x_train'] = treatment_candidates_df.apply(apply_get_labelled_data, all_conditions_set=all_conditions_set, all_treatments_set=all_treatments_set, axis=1)
	treatment_candidates_df['score'] = treatment_candidates_df.apply(apply_score, model=model, axis=1)
	treatment_candidates_df.to_sql('raw_treatment_recs_staging_4', engine, schema='annotation', if_exists='append', index=False, dtype={'sentence_tuples' : sqla.types.JSON, 'concept_arr' : sqla.types.JSON})
	u.pprint("write")


def get_labelled_data_sentence(sentence, condition_id, tx_id, conditions_set, all_treatments_set):
	final_results = pd.DataFrame()

	sample = [0]*max_words
	
	counter=0
	for index,words in enumerate(sentence['sentence_tuples']):
		## words[1] is the conceptid, words[0] is the word
		# condition 1 -- word is the condition of interest
		if words[1] == condition_id and sample[counter-1] != target_condition_key:
			sample[counter] = target_condition_key
		elif (words[1] in conditions_set) and (sample[counter-1] != generic_condition_key):
			sample[counter] = generic_condition_key
			# - word is generic treatment
		elif (words[1] == tx_id) and (sample[counter-1] != target_treatment_key):
			sample[counter] = target_treatment_key
		elif (words[1] in all_treatments_set) and (sample[counter-1] != generic_treatment_key):
			sample[counter] = generic_treatment_key
			# Now using conceptid if available
		elif words[1] != 0 and (get_word_index(words[1]) != UNK_ind) and get_word_index(words[1]) != sample[counter-1]:
			sample[counter] = get_word_index(words[1])
		elif (words[1] == 0 and (get_word_index(words[0]) != UNK_ind) and get_word_index(words[0]) != sample[counter-1])\
			and words[1] not in conditions_set and words[1] not in all_treatments_set:
			sample[counter] = get_word_index(words[0])
		elif (words[1] == 0) and (get_word_index(words[0]) == UNK_ind) and words[1] not in conditions_set and words[1] not in all_treatments_set:
			sample[counter] = get_word_index(words[0])
		else:
			counter -= 1
		
		counter += 1

		if counter >= max_words-1:
			break
			# while going through, see if conceptid associated then add to dataframe with root index and rel_type index

	return sample


def gen_datasets_top():
	conn,cursor = pg.return_postgres_cursor()
	engine = pg.return_sql_alchemy_engine()

	query = "select min(ver) as ver from annotation.labelled_treatments t1"
	new_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])+1
	old_version = new_version-1
	conditions_set = get_all_conditions_set()
	treatments_set = get_all_treatments_set()


	query = "select count(*) as cnt from annotation.labelled_treatments"
	max_counter = int(pg.return_df_from_query(cursor, query, None, ['cnt'])['cnt'][0])

	query = """
			select t1.id, condition_id, treatment_id, label, ver from annotation.labelled_treatments t1 where ver=%s limit 1
		"""
	labels_df = pg.return_df_from_query(cursor, query, (old_version,), ['id', 'condition_id', 'treatment_id', 'label', 'ver'])
	
	while (len(labels_df) > 0):
		print(labels_df)
		write_sentence_vectors_from_labels(labels_df, conditions_set, treatments_set, engine, cursor)
		query = "UPDATE annotation.labelled_treatments set ver = %s where id = %s"
		cursor.execute(query, (new_version, labels_df['id'].values[0]))
		cursor.connection.commit()

		query = """
			select t1.id, condition_id, treatment_id, label, ver from annotation.labelled_treatments t1 where ver=%s limit 1
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

def write_sentence_vectors_from_labels(labels_df, conditions_set, treatments_set, engine, cursor):

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
							,t1.conceptid as treatment_id
							,t2.condition_id
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
		
		if len(sentences_df.index) > 0:
			sentences_df['x_train'] = sentences_df.apply(apply_get_labelled_data, all_conditions_set=conditions_set, all_treatments_set=treatments_set, axis=1)
			sentences_df['label'] = item['label']
			sentences_df.to_sql('training_sentences', engine, schema='annotation', if_exists='append', index=False, dtype={'sentence_tuples' : sqla.types.JSON, 'x_train' : sqla.types.JSON, 'sentence' : sqla.types.Text})

		

if __name__ == "__main__":
	# conn, cursor = pg.return_postgres_cursor()
	# print(train_data_generator(10, cursor))
	# train_with_word2vec()
	parallel_treatment_recategorization_top('model-02.hdf5')
	# gen_datasets_top()