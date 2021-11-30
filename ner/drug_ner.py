import spacy
from spacy.util import minibatch, compounding
import random
import sys
import pandas as pd
sys.path.append('utilities')
import pglib as pg
from tqdm import tqdm
import multiprocessing as mp
import sqlalchemy as sqla
import json
from pathlib import Path

label = "DRUG"

def gen_dataset_worker(input):
	for func,args in iter(input.get, 'STOP'):
		gen_dataset_calculate(func, args)


def gen_dataset_calculate(func, args):
	func(*args)


def gen_ner_data_top():
	conn,cursor = pg.return_postgres_cursor()
	all_drugs = get_all_drugs()
	query = "select min(ver) from spacy.drug_ner_sentences"
	new_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])+1
	old_version = new_version-1
	curr_version = old_version

	while curr_version != new_version:
		gen_ner_data_bottom(old_version, new_version, all_drugs)
		curr_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])
		
def gen_ner_data_bottom(old_version, new_version, all_drugs):
	number_of_processes = 20

	task_queue = mp.Queue()
	pool = []

	for i in range(number_of_processes):
		p = mp.Process(target=gen_dataset_worker, args=(task_queue,))
		pool.append(p)
		p.start()

	conn,cursor = pg.return_postgres_cursor()

	get_query = """
		select 
			sentence_id
			,sentence_tuples
		from spacy.drug_ner_sentences
		where ver = %s limit 5000
	"""

	ner_sentences_df = pg.return_df_from_query(cursor, get_query, \
				(old_version,), ['sentence_id', 'sentence_tuples'])

	counter = 0

	while (counter < number_of_processes) and (len(ner_sentences_df.index) > 0):
		params = (ner_sentences_df, all_drugs)
		task_queue.put((gen_drug_ner_dataset, params))
		update_query = """
			UPDATE spacy.drug_ner_sentences
			SET ver = %s
			where sentence_id = ANY(%s);
		"""
		cursor.execute(update_query, (new_version, ner_sentences_df['sentence_id'].values.tolist(), ))
		cursor.connection.commit()

		ner_sentences_df = pg.return_df_from_query(cursor, get_query, \
				(old_version,), ['sentence_id', 'sentence_tuples'])

		counter += 1

	for i in range(number_of_processes):
		task_queue.put('STOP')

	for p in pool:
		p.join()

	for p in pool:		
		p.close()

	cursor.close()
	conn.close()

def gen_drug_ner_dataset(ner_sentences_df, all_drugs):

	ner_sentences_dict = ner_sentences_df.to_dict('records')
	res_df = pd.DataFrame()
	for row in ner_sentences_dict:
		sentence_tuples = row['sentence_tuples']
		start_index = 0
		full_sentence = ""
		res_tuple = None
		res_dict = {"entities" : []}
		for item in sentence_tuples:
			full_sentence += item[0]
			full_sentence += " "
			if item[1] in all_drugs:
				end_index = start_index + len(item[0])
				res_dict["entities"].append((start_index, end_index, label))
			start_index += len(item[0])+1
		full_sentence = full_sentence.rstrip()

		res_df = res_df.append(pd.DataFrame([[row['sentence_id'], full_sentence, res_dict, 0]], columns=['sentence_id', 'sentence','train', 'ver']))

	engine = pg.return_sql_alchemy_engine()

	res_df.to_sql('drug_ner_train', engine, schema='spacy', if_exists='append', \
		 index=False, dtype={'train' : sqla.types.JSON})
	engine.dispose()

def get_all_drugs():
	query = "select child_acid as root_acid from snomed2.transitive_closure_acid where parent_acid='250597' "
	conn,cursor = pg.return_postgres_cursor()
	all_drugs = set(pg.return_df_from_query(cursor, query, None, ['root_cid'])['root_cid'].tolist())
	return all_drugs

# def save_train_doc_bin(ver):
# 	conn,cursor = pg.return_postgres_cursor()
# 	get_query = """
# 		select 
# 			sentence_id
# 			,sentence
# 			,train
# 		from spacy.drug_ner_train
# 		where ver = %s limit 400000
# 	"""
# 	ner_sentences_df = pg.return_df_from_query(cursor, get_query, \
# 				(ver,), ['sentence_id', 'sentence', 'train'])

# 	TRAIN_DATA = list(ner_sentences_df[['sentence', 'train']].itertuples(index=False, name=None))
	
# 	nlp = spacy.load("en_core_web_sm")
# 	db = DocBin()
# 	db = convert_train_format(nlp, db, TRAIN_DATA)
# 	db.to_disk("./ner/drug_train.spacy")

# 	update_query = """
# 			UPDATE spacy.drug_ner_train
# 			SET ver = %s
# 			where sentence_id = ANY(%s);
# 	"""
# 	cursor.execute(update_query, (ver+1, ner_sentences_df['sentence_id'].values.tolist(), ))
# 	cursor.connection.commit()

# 	get_query = """
# 		select 
# 			sentence_id
# 			,sentence
# 			,train
# 		from spacy.drug_ner_train
# 		where ver = %s
# 	"""
# 	ner_sentences_df = pg.return_df_from_query(cursor, get_query, \
# 				(ver,), ['sentence_id', 'sentence', 'train'])

# 	VALIDATION_DATA = list(ner_sentences_df[['sentence', 'train']].itertuples(index=False, name=None))

# 	db = DocBin()
# 	db = convert_train_format(nlp, db, VALIDATION_DATA)
# 	db.to_disk("./ner/drug_validation.spacy")

# 	update_query = """
# 			UPDATE spacy.drug_ner_train
# 			SET ver = %s
# 			where sentence_id = ANY(%s);
# 	"""
# 	cursor.execute(update_query, (ver+1, ner_sentences_df['sentence_id'].values.tolist(), ))
# 	cursor.connection.commit()

# 	cursor.close()
# 	conn.close()

# def convert_train_format(nlp, db, TRAIN_DATA):
# 	for text, annot in tqdm(TRAIN_DATA):
# 		doc = nlp.make_doc(text)
# 		ents = []
# 		for start, end, label in annot["entities"]:
# 			span = doc.char_span(start, end, label=label, alignment_mode='contract')
# 			if span is None:
# 				print("skipping entity")
# 			else:
# 				ents.append(span)
# 		doc.ents = ents
# 		db.add(doc)
# 	return db




# print("test")
# gen_ner_data_top()


# nlp = spacy.load("en_core_web_sm")

# ner = nlp.get_pipe('ner')

# conn,cursor = pg.return_postgres_cursor()
# query = "select sentence, train from spacy.drug_ner_train"
# train_df = pg.return_df_from_query(cursor, query, None, ['sentence', 'train'])
# train_data = list(train_df.itertuples(index=False, name=None))


# ner.add_label(label)
# optimizer = nlp.resume_training()
# move_names = list(ner.move_names)
# pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
# other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# with nlp.disable_pipes(*other_pipes):
# 	print("start")
# 	# optimizer = nlp.begin_training()
# 	for itn in range(20):
# 		random.shuffle(train_data)
# 		losses = {}
# 		batches = minibatch(train_data, size=compounding(4., 32., 1.001))
# 		for batch in batches:
# 			texts, annotations = zip(*batch)
# 			nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
# 		print('Losses', losses)
output_dir = Path("./ner/drug_ner_model2/")
# nlp.meta['name'] = "drug_ner"
# nlp.to_disk(output_dir)

nlp = spacy.load(output_dir)
doc = nlp("apap and flubitril for knee arthritis")
for ent in doc.ents:
	print(ent.label_, ent.text)