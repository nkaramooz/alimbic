#TO DO: Clean up code to use new pglib. Remove unnecessary import statements.
# Also add comments!

import pandas as pd
from sqlalchemy import create_engine
from nltk.stem.wordnet import WordNetLemmatizer
from operator import itemgetter
import nltk.data
import numpy as np
import time
import utilities.pglib as pg
import utilities.utils2 as u
import sys
from snomed_annotator import snomed_annotator as ann
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten
from keras.layers import Bidirectional
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import sqlalchemy as sqla
import multiprocessing as mp
import datetime


# index 0 = filler
# index 1 = UNK
spacer_ind = 0
UNK_ind = 1
vocabulary_size = 50000
max_words = 60

# Vocabulary spacer should be 4
vocabulary_spacer = 4
target_treatment_key = vocabulary_size-vocabulary_spacer+3 #49,999
target_condition_key = vocabulary_size-vocabulary_spacer+2 #49,998
generic_condition_key = vocabulary_size-vocabulary_spacer+1 #49,997
generic_treatment_key = vocabulary_size-vocabulary_spacer #49,996

# bounded
b_target_treatment_key_start = target_treatment_key # 49997
b_target_treatment_key_end = vocabulary_size+1 # 50001
b_generic_treatment_key_start = generic_treatment_key # 4995
b_generic_treatment_key_stop = vocabulary_size+2 # 50002

def get_all_concepts_of_interest():
	concepts_of_interest = get_all_conditions_set()
	concepts_of_interest.update(get_all_treatments_set())
	concepts_of_interest.update(get_all_diagnostics_set())
	concepts_of_interest.update(get_all_causes_set())
	concepts_of_interest.update(get_all_outcomes_set())
	concepts_of_interest.update(get_all_statistics_set())
	concepts_of_interest.update(get_all_chemicals_set())
	concepts_of_interest.update(get_all_study_designs_set())
	return concepts_of_interest

def get_all_conditions_set():
	query = """select root_acid from annotation2.concept_types 
		where (rel_type='condition' or rel_type='symptom') 
		and (active=1 or active=3)
	"""
	all_conditions_set = set(pg.return_df_from_query(query, None, ['root_acid'])['root_acid'].tolist())
	return all_conditions_set

def get_all_outcomes_set():
	query = """select root_acid from annotation2.concept_types 
		where (rel_type='outcome') 
		and (active=1 or active=3)
	"""
	all_outcomes_set = set(pg.return_df_from_query(query, None, ['root_acid'])['root_acid'].tolist())
	return all_outcomes_set

def get_all_statistics_set():
	query = """select root_acid from annotation2.concept_types 
		where (rel_type='statistic') 
		and (active=1 or active=3)
	"""
	all_statistics_set = set(pg.return_df_from_query(query, None, ['root_acid'])['root_acid'].tolist())
	return all_statistics_set

def get_all_study_designs_set():
	query = """select root_acid from annotation2.concept_types 
		where (rel_type='study_design') 
		and (active=1 or active=3)
	"""
	all_study_designs_set = set(pg.return_df_from_query(query, None, ['root_acid'])['root_acid'].tolist())
	return all_study_designs_set

def get_all_chemicals_set():
	query = """select root_acid from annotation2.concept_types 
		where (rel_type='chemical') 
		and (active=1 or active=3)
	"""
	all_chemicals_set = set(pg.return_df_from_query(query, None, ['root_acid'])['root_acid'].tolist())
	return all_chemicals_set

def get_all_treatments_set():
	query = """
		select root_acid from annotation2.concept_types 
		where rel_type='treatment' 
		and (active=1 or active=3)
	"""
	all_treatments_set = set(pg.return_df_from_query(query, None, ['root_cid'])['root_cid'].tolist())
	return all_treatments_set

def get_all_anatomy_set():
	query = """
		select root_acid from annotation2.concept_types 
		where rel_type='anatomy' 
		and (active=1 or active=3)
	"""
	all_anatomy_set = set(pg.return_df_from_query(query, None, ['root_cid'])['root_cid'].tolist())
	return all_anatomy_set

def get_all_treatments_with_inactive():
	query = """
		select root_acid from annotation2.concept_types 
		where rel_type='treatment'
	"""
	all_treatments_set = set(pg.return_df_from_query(query, None, ['root_cid'])['root_cid'].tolist())
	return all_treatments_set

def get_all_causes_set():
	query = """
		select root_acid from annotation2.concept_types 
		where rel_type='cause' 
		and (active=1 or active=3)
	"""
	all_cause_set = set(pg.return_df_from_query(query, None, ['root_cid'])['root_cid'].tolist())
	return all_cause_set

def get_all_diagnostics_set():
	query = """
		select root_acid from annotation2.concept_types 
		where rel_type='diagnostic' 
		and (active=1 or active=3) 
	"""
	all_diagnostic_set = set(pg.return_df_from_query(query, None, ['root_cid'])['root_cid'].tolist())
	return all_diagnostic_set

def get_word_index(word):
	query = "select rn from ml2.word_counts_50k where word=%s limit 1"
	ind_df = pg.return_df_from_query(query, (word,), ['rn'])
	
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
		
		query = """
			select 
				id
				,x_train_gen
				,x_train_spec
				,x_train_gen_mask
				,x_train_spec_mask
				,label
			from ml2.train_sentences 
			where ver = %s order by random() limit %s 
			"""
		train_df = pg.return_df_from_query(cursor, query, (curr_version, batch_size), 
			['id', 'x_train_gen', 'x_train_spec', 'x_train_gen_mask', 'x_train_spec_mask', 'label'])

		x_train_gen = train_df['x_train_gen'].tolist()


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

	embedding_size=500
	batch_size = 500
	num_epochs = 40

	model_input_gen = Input(shape=(max_words,))
	model_gen_emb = Embedding(vocabulary_size+1, embedding_size, trainable=True, mask_zero=True)(model_input_gen)
	lstm_model = Bidirectional(LSTM(256, recurrent_dropout=0.4, return_sequences=True))(model_gen_emb)
	lstm_model = Bidirectional(LSTM(500, recurrent_dropout=0.5, return_sequences=True))(lstm_model)
	lstm_model = Bidirectional(LSTM(256, recurrent_dropout=0.4))(lstm_model)
	lstm_model = Dropout(0.3)(lstm_model)
	lstm_model = Dense(256)(lstm_model)
	lstm_model = Dropout(0.3)(lstm_model)
	pred = Dense(1, activation='sigmoid')(lstm_model)
	
	model = Model(inputs=[model_input_gen], outputs=[pred])

	print(model.summary())
	# model.summary(print_fn=lambda x: report.write(x + '\n'))
	# report.write('\n')

	model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

	checkpointer = ModelCheckpoint(filepath='./gen_bidi_500_deep_{epoch:02d}.hdf5', verbose=1)
	log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

	x_val, y_val = get_validation_set(cursor)

	history = model.fit(train_data_generator_v2(batch_size, cursor), \
	 epochs=num_epochs, class_weight={0:1, 1:1.2}, steps_per_epoch =((max_cnt//batch_size)+1),
	  validation_data=(x_val, y_val), callbacks=[checkpointer, tensorboard_callback])
	cursor.close()
	conn.close()

def update_rnn(model_name, max_cnt):
	conn,cursor = pg.return_postgres_cursor()
	embedding_size=500
	batch_size = 300
	num_epochs = 45
	model = load_model(model_name)

	checkpointer = ModelCheckpoint(filepath='./gen_bidi_deep_500_update_{epoch:02d}.hdf5', verbose=1)

	
	history = model.fit(train_data_generator_v2(batch_size, cursor), \
	 epochs=num_epochs, class_weight={0:1, 1:2.2}, steps_per_epoch =((max_cnt//batch_size)+1),
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
	number_of_processes = 5

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
		where ver = %s limit 10000
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

	data = np.asarray(treatment_candidates_df['x_train_gen'].tolist()).astype('float32')
	res = model.predict(data)
	treatment_candidates_df['score'] = res
	treatment_candidates_df.to_sql('treatment_recs_staging_2', engine, schema='ml2', if_exists='append', \
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


# spec == True
# Generates a broader training set where for example
# the model will train that a symptom in a sentence is not 
# a treatment for a condition in the same sentence
# so it should learn features specific to the word of concept
def gen_datasets_mp(new_version, spec):
	number_of_processes = 40
	old_version = new_version-1
	conditions_set = get_all_conditions_set()

	if spec == False:
		treatments_set = get_all_treatments_set() #model did not learn well with using inactive
	else:
		# for specific word + mask dataset, will broaden to include all concepts of interest
		treatments_set = get_all_concepts_of_interest()

	task_queue = mp.Queue()
	pool = []

	for i in range(number_of_processes):
		p = mp.Process(target=gen_dataset_worker, args=(task_queue, conditions_set, treatments_set, spec))
		pool.append(p)
		p.start()

	conn,cursor = pg.return_postgres_cursor()

	# for generic, if broadened to ignore whether active, model not learning well
	if spec == False:
		get_query = """
				select t1.id, condition_acid::text, treatment_acid::text, label, ver 
				from ml2.labelled_treatments t1 where ver=%s and (label=0 or label=1)
				and treatment_acid in 
					(select root_acid from annotation2.concept_types 
						where rel_type='treatment' 
						and (active=1 or active=3)) 
				limit %s
			"""
	else:
		get_query = """
				select t1.id, condition_acid::text, treatment_acid::text, label, ver 
				from ml2.labelled_treatments t1 where ver=%s and (label=0 or label=1)
				limit %s
			"""
	labels_df = pg.return_df_from_query(cursor, get_query, (old_version, number_of_processes),
		['id', 'condition_acid', 'treatment_acid', 'label', 'ver'])
	

	while len(labels_df.index) > 0:
		for index,item in labels_df.iterrows():
			params = (item['condition_acid'], item['treatment_acid'], item['label'])
			task_queue.put((gen_datasets_mp_bottom, params))

		update_query = "UPDATE ml2.labelled_treatments set ver = %s where id in %s"
		cursor.execute(update_query, (new_version, tuple(labels_df['id'].values.tolist())))
		cursor.connection.commit()

		labels_df = pg.return_df_from_query(cursor, get_query, (old_version, number_of_processes),
			['id', 'condition_acid', 'treatment_acid', 'label', 'ver'])
		
	for i in range(number_of_processes):
		task_queue.put('STOP')

	for p in pool:
		p.join()

	for p in pool:
		p.close()

	cursor.close()
	conn.close()


def gen_datasets_mp_bottom(condition_acid, treatment_acid, label, conditions_set, treatments_set, spec):
	conn,cursor = pg.return_postgres_cursor()
	sentences_df = pd.DataFrame()

	if condition_acid == '%':
		tx_id_arr = [treatment_acid]
		tx_id_arr.extend(ann.get_children(treatment_acid, cursor))
		
		sentences_query = """
						select
							t4.sentence_tuples
							,t2.condition_acid
							,t3.treatment_acid
							,t1.sentence_id
							,t1.section_ind
							,t1.pmid
						from pubmed.sentence_concept_arr_2 t1
						join (select root_acid as condition_acid 
							from annotation2.concept_types 
							where rel_type='condition' or rel_type='symptom' or rel_type='cause'
							and (active=1 or active=3)) t2
							on t2.condition_acid = ANY(t1.concept_arr::text[])
						join (select acid as treatment_acid from annotation2.downstream_root_cid where acid in %s) t3
							on t3.treatment_acid = ANY(t1.concept_arr::text[])
						join pubmed.sentence_tuples_2 t4
							on t1.sentence_id = t4.sentence_id
					"""
		sentences_df = pg.return_df_from_query(cursor, sentences_query, (tuple(tx_id_arr),), \
				['sentence_tuples', 'condition_acid', 'treatment_acid', 'sentence_id', 'section_ind', 'pmid'])
		sentences_df['og_condition_acid'] = '%'
		sentences_df['og_treatment_acid'] = treatment_acid
		print("finished query")
		print("len of wildcard: " + str(len(sentences_df)))
	elif treatment_acid != '%':
		condition_id_arr = [condition_acid]
		condition_id_arr.extend(ann.get_children(condition_acid, cursor))
		tx_id_arr = [treatment_acid]
		tx_id_arr.extend(ann.get_children(treatment_acid, cursor))
		sentences_query = """
				select
					t4.sentence_tuples
					,t2.condition_acid
					,t3.treatment_acid
					,t1.sentence_id
					,t1.section_ind
					,t1.pmid
				from pubmed.sentence_concept_arr_2 t1
				join (select acid as condition_acid from annotation2.downstream_root_cid where acid in %s) t2
					on t2.condition_acid = ANY(t1.concept_arr::text[])
				join (select acid as treatment_acid from annotation2.downstream_root_cid where acid in %s) t3
					on t3.treatment_acid = ANY(t1.concept_arr::text[])
				join pubmed.sentence_tuples_2 t4
					on t1.sentence_id = t4.sentence_id
			"""
		sentences_df = pg.return_df_from_query(cursor, sentences_query, (tuple(condition_id_arr), tuple(tx_id_arr)), \
				['sentence_tuples', 'condition_acid', 'treatment_acid', 'sentence_id', 'section_ind', 'pmid'])
		sentences_df['og_condition_acid'] = condition_acid
		sentences_df['og_treatment_acid'] = treatment_acid

	cursor.close()
	conn.close()

	if len(sentences_df.index) > 0:
		sentences_df['label'] = label
		write_sentence_vectors_from_labels(sentences_df, conditions_set, treatments_set, spec)


def gen_treatment_data_top():
	conn,cursor = pg.return_postgres_cursor()
	
	all_conditions_set = get_all_conditions_set() 
	all_treatments_set = get_all_treatments_set()


	query = "select min(ver) from ml2.treatment_candidates_2"
	new_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])+1
	old_version = new_version-1
	curr_version = old_version

	while curr_version != new_version:
		gen_treatment_data_bottom(old_version, new_version, all_conditions_set, all_treatments_set, False)
		curr_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])
	

def gen_treatment_data_bottom(old_version, new_version, conditions_set, treatments_set, spec):
	number_of_processes = 35

	task_queue = mp.Queue()
	pool = []

	for i in range(number_of_processes):
		p = mp.Process(target=gen_dataset_worker, args=(task_queue, conditions_set, treatments_set, spec))
		pool.append(p)
		p.start()

	conn,cursor = pg.return_postgres_cursor()

	get_query = """
		select 
			entry_id
			,condition_acid
			,treatment_acid
			,sentence_tuples
		from ml2.treatment_candidates_2

		where ver = %s limit 1000"""

	treatment_candidates_df = pg.return_df_from_query(cursor, get_query, \
				(old_version,), ['entry_id', 'condition_acid', 'treatment_acid', 'sentence_tuples'])

	counter = 0

	while (counter < number_of_processes) and (len(treatment_candidates_df.index) > 0):
		params = (treatment_candidates_df,)
		task_queue.put((gen_treatment_dataset, params))
		update_query = """
			set schema 'ml2';
			UPDATE treatment_candidates_2
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

	for p in pool:		
		p.close()

	cursor.close()
	conn.close()

def gen_treatment_dataset(treatment_candidates_df, all_conditions_set, all_treatments_set, spec):
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

def gen_dataset_worker(input, conditions_set, treatments_set, spec):
	for func,args in iter(input.get, 'STOP'):
		gen_dataset_calculate(func, args, conditions_set, treatments_set, spec)

def gen_dataset_calculate(func, args, conditions_set, treatments_set, spec):
	func(*args, conditions_set, treatments_set, spec)

def write_sentence_vectors_from_labels(sentences_df, conditions_set, treatments_set, spec):
	conn,cursor = pg.return_postgres_cursor()
	engine = pg.return_sql_alchemy_engine()

	if len(sentences_df.index) > 0:
		sentences_df = sentences_df.apply(apply_get_generic_labelled_data, \
			all_conditions_set=conditions_set, all_treatments_set=treatments_set, axis=1)

		sentences_df['ver'] = 0
		sentences_df = sentences_df[['sentence_id', 'sentence_tuples', 'condition_acid', 'treatment_acid',\
		 'og_condition_acid', 'og_treatment_acid', 'x_train_gen', 'x_train_gen_mask', \
		 'x_train_spec_mask','x_train_spec', 'label', 'ver']]


		if spec == False: 
			table_name = 'training_sentences_staging'
		else:
			table_name = 'spec_training_sentences_staging'

		sentences_df.to_sql(table_name, engine, schema='ml2', if_exists='append', index=False, \
			dtype={'sentence_tuples' : sqla.types.JSON, 'x_train_gen' : sqla.types.JSON, 'x_train_gen_mask' : sqla.types.JSON, \
				'x_train_spec_mask' : sqla.types.JSON, 'x_train_spec' : sqla.types.JSON, 'sentence' : sqla.types.Text})
			
	cursor.close()
	conn.close()
	engine.dispose()

def analyze_sentence(model_name, sentence):
	conn,cursor = pg.return_postgres_cursor()
	term = ann.clean_text(sentence)
	lmtzr = WordNetLemmatizer()
	all_words = ann.get_all_words_list(term, lmtzr)
	
	
	item = pd.DataFrame([[term, 'title', 0, 0]], columns=['line', 'section', 'section_ind', 'ln_num'])
	cache = ann.get_cache(all_words_list=all_words, case_sensitive=True, \
			check_pos=False, spellcheck_threshold=100, lmtzr=lmtzr)

	sentence_annotations_df, sentence_tuples_df, sentence_concept_arr_df = ann.annotate_text_not_parallel(sentences_df=item, cache=cache, \
			case_sensitive=True, check_pos=False, bool_acr_check=False,\
			spellcheck_threshold=100, \
			write_sentences=True, lmtzr=lmtzr)

	model = load_model(model_name)

	all_conditions_set = get_all_conditions_set()
	all_treatments_set = get_all_treatments_set()

	relevant_conditions = {}
	for ind,word in enumerate(sentence_tuples_df['sentence_tuples'][0]):
		if word[1] in all_conditions_set and word[1] not in relevant_conditions:
			condition_name = ann.get_preferred_concept_names(word[1], cursor)
			relevant_conditions[word[1]] = condition_name

	final_res = {}

	for condition_id in relevant_conditions.keys():
		for ind,word in enumerate(sentence_tuples_df['sentence_tuples'][0]):
			if word[1] == condition_id:
				continue
			elif word[1] != 0 and word[1] in all_treatments_set:
				sample_gen,sample_spec, mask = get_labelled_data_sentence_generic_v2_custom(sentence_tuples_df, condition_id, word[1], \
					all_conditions_set, all_treatments_set)
				sample_gen_arr = np.array([sample_gen])
	
				res = model.predict(sample_gen_arr)
				res = float(model.predict([sample_gen_arr]))
				if relevant_conditions[condition_id] in final_res.keys():
					final_res[relevant_conditions[condition_id]].append((word[0], word[1], res))
				else:
					final_res[relevant_conditions[condition_id]] = [(word[0], word[1], res)]

	return final_res

def print_contingency(model_name, validation):
	conn, cursor = pg.return_postgres_cursor()
	model = load_model(model_name)

	if validation:
		curr_version = int(pg.return_df_from_query(cursor, "select min(ver) from ml2.validation_sentences", \
				None, ['ver'])['ver'][0])
	else: 
		curr_version = int(pg.return_df_from_query(cursor, "select min(ver) from ml2.test_sentences", \
					None, ['ver'])['ver'][0])
	new_version = curr_version + 1

	# should be OK to load into memory
	if validation:
		testing_query = """
			select 
				id
				,sentence_id
				,sentence_tuples
				,condition_acid
				,treatment_acid
				,x_train_gen
				,x_train_gen_mask
				,label 
			from ml2.validation_sentences where ver=%s
		"""
		sentences_df = pg.return_df_from_query(cursor, testing_query, (curr_version,), \
			['id', 'sentence_id','sentence_tuples','condition_acid', \
			'treatment_acid', 'x_train_gen','x_train_gen_mask', 'label'])
	else:
		testing_query = """
			select 
				id
				,sentence_id
				,sentence_tuples
				,condition_acid
				,treatment_acid
				,x_train_gen
				,x_train_gen_mask
				,label 
			from ml2.test_sentences where ver=%s
		"""
		sentences_df = pg.return_df_from_query(cursor, testing_query, (curr_version,), \
			['id', 'sentence_id','sentence_tuples','condition_acid', \
			'treatment_acid', 'x_train_gen','x_train_gen_mask', 'label'])
	
	zero_zero = 0
	zero_one = 0
	one_zero = 0
	one_one = 0
	sentences_dict = sentences_df.to_dict('records')
	
	for item in sentences_dict:
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

	print(model_name)
	print("precision : " + str((one_one/(one_one+zero_one))))
	print("recall : " + str((one_one/(one_one + one_zero))))
	print("label 1, res 1: " + str(one_one))
	print("label 1, res 0: " + str(one_zero))
	print("label 0, res 0: " + str(zero_zero))
	print("label 0, res 1: " + str(zero_one))

	# if (one_one + one_zero) != 0:
	# 	sens = (one_one) / (one_one + one_zero)
	# 	print("sensitivity : " + str(sens))
	# if (zero_one + zero_zero) != 0:
	# 	spec = zero_zero / (zero_one + zero_zero)
	# 	print("specificity : " + str(spec))

	cursor.close()
	conn.close()

def get_validation_set(cursor):
	query = "select x_train_gen, label from ml2.train_sentences"
	train_df = pg.return_df_from_query(cursor, query, None, ['x_train_gen', 'label'])
	x_train_gen = train_df['x_train_gen'].tolist()
	y_train = train_df['label'].tolist()
	return (np.asarray(x_train_gen), np.asarray(y_train))


if __name__ == "__main__":

	sentence = "acute interstitial nephritis was treated with nitroglycerin"
	sentence = sentence.lower()
	model_name = 'gen_bidi_500_deep_18.hdf5'
	condition_id = '10603'
	print(analyze_sentence(model_name, sentence))
