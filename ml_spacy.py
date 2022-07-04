import spacy
from spacy import displacy
import ml2
import utilities.pglib as pg
import sqlalchemy as sqla
import multiprocessing as mp
import pandas as pd
from spacy.tokens import DocBin
from typing import Callable, Iterator, List
from spacy.training import Example
from spacy.language import Language
from tqdm import tqdm

# conditions_set = ml2.get_all_conditions_set()
# treatments_set = ml2.get_all_treatments_set()
# diagnostics_set = ml2.get_all_diagnostics_set()
# causes_set = ml2.get_all_causes_set()
# outcomes_set = ml2.get_all_outcomes_set()
# statistics_set = ml2.get_all_statistics_set()

def gen_dataset_worker(input):
	for func,args in iter(input.get, 'STOP'):
		gen_dataset_calculate(func, args)

def gen_dataset_calculate(func, args):
	func(*args)

def gen_ner_data_top():
	conn,cursor = pg.return_postgres_cursor()
	query = "select min(ver) from spacy.concept_ner_tuples"
	new_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])+1
	old_version = new_version-1
	curr_version = old_version

	while curr_version != new_version:
		gen_ner_data_bottom(old_version, new_version)
		curr_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])
		
def gen_ner_data_bottom(old_version, new_version):
	number_of_processes = 40

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
			,rand
		from spacy.concept_ner_tuples
		where ver = %s limit 5000
	"""

	ner_sentences_df = pg.return_df_from_query(cursor, get_query, \
				(old_version,), ['sentence_id', 'sentence_tuples', 'rand'])

	counter = 0

	while (counter < number_of_processes) and (len(ner_sentences_df.index) > 0):
		params = (ner_sentences_df,)
		task_queue.put((gen_ner_dataset, params))
		update_query = """
			UPDATE spacy.concept_ner_tuples
			SET ver = %s
			where sentence_id = ANY(%s);
		"""
		cursor.execute(update_query, (new_version, ner_sentences_df['sentence_id'].values.tolist(), ))
		cursor.connection.commit()

		ner_sentences_df = pg.return_df_from_query(cursor, get_query, \
				(old_version,), ['sentence_id', 'sentence_tuples', 'rand'])

		counter += 1

	for i in range(number_of_processes):
		task_queue.put('STOP')

	for p in pool:
		p.join()

	for p in pool:		
		p.close()

	cursor.close()
	conn.close()

def gen_ner_dataset(ner_sentences_df):
	ner_sentences_dict = ner_sentences_df.to_dict('records')
	res_df = pd.DataFrame()

	for row in ner_sentences_dict:
		sentence_tuples = row['sentence_tuples']

		start_index = 0
		current_a_cid = None
		current_type = None
		concept_index = None
		full_sentence = ""
		res_tuple = None
		
		labels = []
		for item in sentence_tuples:
			full_sentence += item[0]
			full_sentence += " "

			if current_a_cid != None and item[1] != current_a_cid:
				# File to res_dict
				labels.append((concept_index, start_index-1, current_type))
				current_a_cid = None
				current_type = None
				concept_index = None

			if item[1] in conditions_set and item[1] != current_a_cid:
				concept_index = start_index
				current_a_cid = item[1]
				current_type = 'CONDITION'
			elif item[1] in treatments_set and item[1] != current_a_cid:
				concept_index = start_index
				current_a_cid = item[1]
				current_type = 'TREATMENT'
			elif item[1] in diagnostics_set and item[1] != current_a_cid:
				concept_index = start_index
				current_a_cid = item[1]
				current_type = 'DIAGNOSTIC'
			elif item[1] in causes_set and item[1] != current_a_cid:
				concept_index = start_index
				current_a_cid = item[1]
				current_type = 'ORGANISM'
			elif item[1] in outcomes_set and item[1] != current_a_cid:
				concept_index = start_index
				current_a_cid = item[1]
				current_type = 'OUTCOME'
			elif item[1] in statistics_set and item[1] != current_a_cid:
				concept_index = start_index
				current_a_cid = item[1]
				current_type = 'STATISTIC'

			start_index += len(item[0])+1
		if current_a_cid != None:
			labels.append((concept_index, start_index-1, current_type))

		full_sentence = full_sentence.rstrip()
		if len(labels) != 0:
			res_df = res_df.append(pd.DataFrame([[row['sentence_id'], (full_sentence, labels), row['rand'], 0]], 
				columns=['sentence_id', 'train', 'rand', 'ver']))

	engine = pg.return_sql_alchemy_engine()

	res_df.to_sql('concept_ner_all', engine, schema='spacy', if_exists='append', \
		 index=False, dtype={'train' : sqla.types.JSON})
	engine.dispose()

# Spacy requires all labels to be initialized first
# Need to get all labels in a small training set to feed
# to spacy in the command line
# This is a rather inelegant waay 
def save_label_set():
	label_list = ['CONDITION', 'TREATMENT', 'DIAGNOSTIC', 'ORGANISM', 'OUTCOME', 'STATISTIC']
	nlp = spacy.load("en_core_web_trf")
	db = DocBin()
	conn,cursor = pg.return_postgres_cursor()
	while len(label_list) > 0:
		query = """
			select 
				sentence_id
				,train
			from spacy.concept_ner_train
			limit 500
		"""
		train_df = pg.return_df_from_query(cursor, query, None, 
			['sentence_id', 'train'])

		for row in train_df.to_dict('records'):
			doc = nlp(row['train'][0])
			ents = []
			for item in row['train'][1]:
				span = doc.char_span(item[0], item[1], label=item[2])
				ents.append(span)
				if item[2] in label_list:
					label_list.remove(item[2])
			doc.ents = ents
		db.add(doc)

		update_query = """
			UPDATE spacy.concept_ner_train
			SET ver = %s
			where sentence_id = ANY(%s);
		"""
		cursor.execute(update_query, (1, train_df['sentence_id'].values.tolist(), ))
		cursor.connection.commit()

	cursor.close()
	conn.close()
	db.to_disk("./labels.spacy")


def get_data_df(is_train):
	conn,cursor = pg.return_postgres_cursor()
	if is_train:
		query = """
			select 
				sentence_id
				,train
			from spacy.concept_ner_train
			where ver=0
			limit 200000
		"""
		data_df = pg.return_df_from_query(cursor, query, None, 
			['sentence_id', 'train'])

		query = """
			UPDATE spacy.concept_ner_train
			SET ver=1 where sentence_id = ANY(%s)
		"""
		sentence_ids = data_df['sentence_id'].tolist()
		cursor.execute(query, (sentence_ids,))
		cursor.connection.commit()
		conn.close()
		cursor.close()
		return data_df
	else:
		query = """
			select 
				sentence_id
				,train
			from spacy.concept_ner_validation
		"""
		data_df = pg.return_df_from_query(cursor, query, None, 
			['sentence_id', 'train'])
		conn.close()
		cursor.close()
		return data_df

def prepare_train_data(is_train):
	print("start load")
	nlp = spacy.load("en_core_web_trf")
	print("end load")
	db = DocBin()
	data_df = get_data_df(is_train)
	print("starting iterator")
	for row in tqdm(data_df.to_dict('records')):
		doc = nlp(row['train'][0])
		ents = []
		for item in row['train'][1]:
			span = doc.char_span(item[0], item[1], label=item[2])
			ents.append(span)
		doc.ents = ents
		db.add(doc)
	if is_train:
		db.to_disk("./train_2.spacy")
	else:
		db.to_disk("./validation.spacy")


# @util.registry.readers("corpus_generator")
# def create_corpus_generator(limit: int = -1) -> Callable[[Language], Iterable[Example]]:
# 	return CorpusGenerator(limit)

# class CorpusGenerator:
# 	def __init__(self, limit):
# 		self.limit = limit

# 	def __call__(self, nlp: Language) -> Iterator[Example]:


if __name__ == "__main__":
	# save_label_set()
	# gen_ner_data_top()
	prepare_train_data(is_train=True)
	# nlp = spacy.load("en_core_web_trf")
	# sentences = [["this is a test"], ['this is another test']]
	# print(nlp(sentences))

	# nlp = spacy.load("./output/model-best")
	# contextualSpellCheck.add_to_pipe(nlp)
	# doc = nlp("Flubiodarone and metroprolol for patients with atria arrhythmia on hemodialysis")

	# displacy.serve(doc, style='ent')