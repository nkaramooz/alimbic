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
from negspacy.negation import Negex
from spacy import util
from spacy.kb import KnowledgeBase
import utilities.utils2 as u
from snomed_annotator2 import clean_text

conditions_set = ml2.get_all_conditions_set()
treatments_set = ml2.get_all_treatments_set()
diagnostics_set = ml2.get_all_diagnostics_set()
causes_set = ml2.get_all_causes_set()
outcomes_set = ml2.get_all_outcomes_set()
statistics_set = ml2.get_all_statistics_set()
chemicals_set = ml2.get_all_chemicals_set()
study_designs_set = ml2.get_all_study_designs_set()

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

	cursor.close()
	conn.close()
		
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
		where ver = %s limit 10000
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
				#IL-10 substrings and 'insulin levels substrings'
				exclusion_strings = ['gene', 'values', 'levels', 'inhibits', 'polymorphisms', 'promoters', 'tracer', 'metabolism', 'concentration', 'meal']
				if (current_a_cid == '16639' or current_a_cid == '18319') \
					and ((any(item[0] in string for string in exclusion_strings) \
					or item[0].isnumeric())):
					pass
				else:
					# File to res_dict
					labels.append((concept_index, start_index-1, current_type))
				current_a_cid = None
				current_type = None
				concept_index = None

			# skip glucose
			if item[1] in treatments_set and item[1] != current_a_cid and item[1] != '343548':
				concept_index = start_index
				current_a_cid = item[1]
				current_type = 'TREATMENT'
			elif (item[1] in conditions_set and item[1] != current_a_cid) or \
				(item[1] in causes_set and item[1] != current_a_cid):
				concept_index = start_index
				current_a_cid = item[1]
				current_type = 'CONDITION'
			elif item[1] in diagnostics_set and item[1] != current_a_cid:
				concept_index = start_index
				current_a_cid = item[1]
				current_type = 'DIAGNOSTIC'
			elif item[1] in study_designs_set and item[1] != current_a_cid:
				concept_index = start_index 
				current_a_cid = item[1]
				current_type = "STUDY_DESIGN"
			elif item[1] in outcomes_set and item[1] != current_a_cid:
				concept_index = start_index
				current_a_cid = item[1]
				current_type = 'OUTCOME'
			elif item[1] in statistics_set and item[1] != current_a_cid:
				concept_index = start_index
				current_a_cid = item[1]
				current_type = 'STATISTIC'
			elif item[1] in chemicals_set and item[1] != current_a_cid:
				concept_index = start_index
				current_a_cid = item[1]
				current_type = 'CHEMICAL'
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
	label_list = ['CONDITION', 'TREATMENT','DIAGNOSTIC', 'OUTCOME', 'STATISTIC', 'CHEMICAL', 'STUDY_DESIGN']
	nlp = spacy.load("en_core_web_trf")
	db = DocBin()
	conn,cursor = pg.return_postgres_cursor()
	while len(label_list) > 0:
		query = """
			select 
				sentence_id
				,train
			from spacy.concept_ner_train
			order by random() limit 500
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
			order by random()
			limit 300000
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
			limit 10000
		"""
		data_df = pg.return_df_from_query(cursor, query, None, 
			['sentence_id', 'train'])
		conn.close()
		cursor.close()
		return data_df

def get_validation_data():
	nlp = spacy.load("en_core_web_trf")
	db = DocBin()
	data_df = get_data_df(False)
	print("starting iterator")
	for row in tqdm(data_df.to_dict('records')):
		doc = nlp(row['train'][0])
		ents = []
		for item in row['train'][1]:
			span = doc.char_span(item[0], item[1], label=item[2])
			ents.append(span)
		doc.ents = ents
		db.add(doc)

	db.to_disk("./validation.spacy")

class DataGenerator():

	def __init__(self, is_train):
		conn,cursor = pg.return_postgres_cursor()
		if is_train:
			self.table = "spacy.concept_ner_train"
		else:
			self.table = "spacy.concept_ner_validation"
		query = "select min(ver) as ver from %s" % self.table
		self.ver = int(pg.return_df_from_query(cursor, query, None, ["ver"])["ver"][0])
		self.current_df = self._get_fresh_df(cursor)
		self.cursor = cursor

	def has_next(self):
		if len(self.current_df) == 0:
			if self._is_postgres_empty(self.cursor):
				return False
			else:
				self.current_df = self._get_fresh_df(self.cursor)
				return True
		else:
			return True

	def get_next(self):
		new_row = self.current_df.iloc[0]
		self.current_df.drop(index=self.current_df.index[0], 
			axis=0, inplace=True)
		query = "UPDATE %s " % self.table
		query += " SET ver=%s where sentence_id = %s"
		sentence_id = new_row['sentence_id']
		self.cursor.execute(query, (self.ver+1, sentence_id))
		self.cursor.connection.commit()
		return new_row

	def _get_fresh_df(self, cursor):
		query = "select sentence_id, train from %s " % self.table
		query += " where ver=%s order by random() limit 300000"
		data_df = pg.return_df_from_query(cursor, query, (self.ver,), 
			['sentence_id', 'train'])

		return data_df

	def _is_postgres_empty(self, cursor):
		query = "select min(ver) as ver from %s " % self.table
		new_ver = pg.return_df_from_query(cursor, query, None, ["ver"])["ver"][0]
		if new_ver != self.ver:
			return True
		else:
			return False

@util.registry.readers("corpus_generator")
def stream_data(augmenter, gold_preproc, limit, max_length) -> Callable[[Language], Iterator[Example]]:
	data_generator = DataGenerator(is_train=True)
	def generate_stream(nlp):
		while data_generator.has_next():
			row = data_generator.get_next()
			sentence = row['train'][0]
			doc = nlp.make_doc(sentence)
			entities = row['train'][1]
			example = Example.from_dict(doc, {"entities" : entities})
			yield example

	return generate_stream

# @util.registry.readers("validation_generator")
# def stream_data(augmenter,gold_preproc, limit, max_length) -> Callable[[Language], Iterator[Example]]:
# 	data_generator = DataGenerator(is_train=False)
# 	def generate_validation_stream(nlp):
# 		while data_generator.has_next():
# 			row = data_generator.get_next()
# 			sentence = row['train'][0]
# 			doc = nlp.make_doc(sentence)
# 			entities = row['train'][1]
# 			example = Example.from_dict(doc, {"entities" : entities})
# 			yield example

# 	return generate_validation_stream


if __name__ == "__main__":
	print("main")
	# gen_ner_data_top()
	# save_label_set()
	# get_validation_data()

	nlp = spacy.load("/home/nkaramooz/Documents/alimbic/web_lg_output/model-last")
	# nlp = spacy.load("en_core_web_lg")
	# nlp = spacy.load("en_core_web_trf")
	# doc = nlp("Roy Emerson")
	# print(doc.ents)
	# print(len(doc.vector))
	# print(len(doc.vector[0]))
	# print(len(doc._.trf_data.tensors[-1][0]))
	# print(doc._.trf_data.tensors[-1])

	
	# kb.to_disk("/home/nkaramooz/Documents/alimbic/kb")

	# kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=768)
	# kb.from_disk("/home/nkaramooz/Documents/alimbic/kb")

	# print(f"Candidate for CHF worsened heart failure : {[c.entity_ for c in kb.get_alias_candidates('CHF patients')]}")
	#### need to create list of potential entities.
	### need list of prior probabilities for each entity.
	# full name resolves with 100% probability
	# for a_cid, name in names_dict.items():
	# 	kb.add_alias(alias=name, entities=[a_cid], probabilities=[1])
	# print(get_kb_aliases())
	# print(agg_df.iloc[0][1])
	# cnt_df = get_concept_counts(agg_df.iloc[0][1])
	# print(cnt_df)
	# print(get_total_concept_counts())

	# qids = name_dict.keys()
	# kb.add_alias()	
	# probs = []
	# kb.add_alias(alias="Emerson", entities=qids, probabilities=probs)
	# sentence = "1,3-benzodioxolyl-N-methylbutanamine used for treating cancer."
	# sentence = "Phase 2 and biomarker study of trebananib, an angiopoietin-blocking peptibody, with and without bevacizumab for patients with recurrent glioblastoma multiforme (GBM)."
	# sentence = "SARS-CoV-2 pneumonia is associated with watery diarrhea and is treated with remdesivir and flubitril."
	# sentence = "C. Diff treated with rifaximin, flubitril, and amiodrarone."
	sentence = "Patients with ankyalosing spondylitis were treated with spironolactone and sacubitril/valsartan in a retrospective case series."
	# sentence = "Postacute/Long COVID in Pediatrics."
	# nlp = spacy.load("/home/nkaramooz/Documents/alimbic/output_3/model-best")
	nlp.add_pipe('sentencizer')
	nlp.add_pipe("negex", config={"ent_types" : ["CONDITION", "TREATMENT"]})
	doc = nlp(sentence)
	for e in doc.ents:
		print(e.text, e._.negex)
	displacy.serve(doc, style='ent')

