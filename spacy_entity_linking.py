import spacy
from spacy import displacy
import utilities.pglib as pg
import sqlalchemy as sqla
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from negspacy.negation import Negex
from spacy import util
from spacy.kb import KnowledgeBase
import utilities.utils2 as u
import math
from spacy.training import Example
from spacy.ml.models import load_kb
from spacy.util import minibatch, compounding
import random
import ml2
import es_kb as elastic_kb
from typing import Callable, Iterator, Iterable, List
from spacy.kb import Candidate
from snomed_annotator2 import clean_text

CONCEPTS_OF_INTEREST = ml2.get_all_concepts_of_interest()
es = u.get_es_client()

def gen_dataset_worker(input):
	for func,args in iter(input.get, 'STOP'):
		gen_dataset_calculate(func, args)


def gen_dataset_calculate(func, args):
	func(*args)


def gen_linking_data_top():
	conn,cursor = pg.return_postgres_cursor()
	query = "select min(ver) from spacy.concept_ner_tuples"
	new_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])+1
	old_version = new_version-1
	curr_version = old_version

	while curr_version != new_version:
		gen_linking_data_bottom(old_version, new_version)
		curr_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])
	cursor.close()
	conn.close()
		

def gen_linking_data_bottom(old_version, new_version):
	number_of_processes = 48

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
		where ver = %s limit 1000
	"""

	linking_sentences_df = pg.return_df_from_query(cursor, get_query, \
				(old_version,), ['sentence_id', 'sentence_tuples', 'rand'])

	counter = 0

	while (counter < number_of_processes) and (len(linking_sentences_df.index) > 0):
		params = (linking_sentences_df,)
		task_queue.put((gen_linking_dataset, params))
		update_query = """
			UPDATE spacy.concept_ner_tuples
			SET ver = %s
			where sentence_id = ANY(%s);
		"""
		cursor.execute(update_query, (new_version, linking_sentences_df['sentence_id'].values.tolist(), ))
		cursor.connection.commit()

		linking_sentences_df = pg.return_df_from_query(cursor, get_query, \
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


def gen_linking_dataset(linking_sentences_df):
	linking_sentences_dict = linking_sentences_df.to_dict('records')
	res_df = pd.DataFrame()

	for row in linking_sentences_dict:
		sentence_tuples = row['sentence_tuples']
		
		start_index = 0
		current_a_cid = None
		concept_index = None
		full_sentence = ""
		hash_dict = {}
		hash_dict['links'] = []

		for item in sentence_tuples:
			full_sentence += item[0]
			full_sentence += " "

			if current_a_cid != None and item[1] != current_a_cid:
				hash_dict['links'].append({'tuple' : (concept_index, start_index-1), 'entity' : current_a_cid})

				current_a_cid = None
				current_type = None
				concept_index = None

			if item[1] != 0 and item[1] != current_a_cid and item[1] in CONCEPTS_OF_INTEREST:
				concept_index = start_index 
				current_a_cid = item[1]

			start_index += len(item[0])+1

		if current_a_cid != None:
			hash_dict['links'].append({'tuple' : (concept_index, start_index-1), 'entity' : current_a_cid})

		full_sentence = full_sentence.rstrip()

		if len(hash_dict['links']) > 0:
			res_df = res_df.append(pd.DataFrame([[row['sentence_id'], (full_sentence, hash_dict), row['rand'], 0]], 
				columns=['sentence_id', 'train', 'rand', 'ver']))

	engine = pg.return_sql_alchemy_engine()

	res_df.to_sql('entity_linking_all', engine, schema='spacy', if_exists='append', \
		 index=False, dtype={'train' : sqla.types.JSON})
	engine.dispose()


def get_kb_primary_entities():
	names = dict()
	descriptions = dict()
	counts = dict()

	conn,cursor = pg.return_postgres_cursor()

	query = """
		select 
			a_cid
			,term
			,description
			,cnt
		from spacy.spacy_primary_entities
	"""

	entities_df = pg.return_df_from_query(cursor, query, None, 
		['a_cid', 'term', 'description', 'cnt'])

	for row in tqdm(entities_df.to_dict('records')):
		names[row['a_cid']] = row['term']
		descriptions[row['a_cid']] = row['description']
		counts[row['a_cid']] = row['cnt']

	cursor.close()
	conn.close()
	
	return names, descriptions, counts


def get_concept_counts(cursor, concept_arr):
	query = """ 
		select t1.a_cid, case when t2.cnt is null then 0 else t2.cnt end as cnt 
		from (select unnest(%s) as a_cid) t1 
		left join annotation2.concept_counts t2 
		on t1.a_cid=t2.concept
	"""
	cnt_df = pg.return_df_from_query(cursor, query, (concept_arr,), ['a_cid', 'cnt'])
	return cnt_df


def load_kb_primary_entities(kb, nlp):
	names_dict, desc_dict, counts = get_kb_primary_entities()

	for a_cid, desc in tqdm(desc_dict.items()):
		desc_doc = nlp(desc)
		# desc_enc = desc_doc._.trf_data.tensors[-1][0] use for transformer model
		desc_enc = desc_doc.vector
		kb.add_entity(entity=a_cid, entity_vector=desc_enc, freq=counts[a_cid])

	return kb


def get_kb_aliases():
	conn,cursor = pg.return_postgres_cursor()
	query = "select term, a_cid_aggs from spacy.concept_alias_aggregates"
	alias_df = pg.return_df_from_query(cursor, query, None, ['term', 'a_cid_aggs'])
	cursor.close()
	conn.close()
	return alias_df

def load_kb_aliases(kb):
	alias_df = get_kb_aliases()
	conn, cursor = pg.return_postgres_cursor()
	es = u.get_es_client()
	for row in tqdm(alias_df.to_dict('records')):
		term = row['term']
		terms_list = elastic_kb.get_training_candidate_terms(es, term)
		term = clean_text(term)
		agg = []
		for item in terms_list:
			agg.extend(elastic_kb.get_exact_candidate_list(es, item))
		agg = list(set(agg))
		if term == 'Myocardial infarction':
			print(term, agg)

		if len(agg) == 1:
			kb.add_alias(alias=term, entities=agg, probabilities=[1])
		else:
			cnt_df = get_concept_counts(cursor, agg)
			# Trying to account for some probability that corpus is not representative
			cnt_df['cnt'] = cnt_df['cnt'] + 1 
			total = cnt_df['cnt'].sum()
			# make sure that total probability equals 1
			if not total:
				prob_arr = [round(1/len(agg),2) for a_cid in agg]
			else:
				prob_arr = [math.floor((float(cnt_df[cnt_df['a_cid'] == a_cid]['cnt'])/total)*100)/100.0 for a_cid in agg]
			kb.add_alias(alias=term, entities=agg, probabilities=prob_arr)
	cursor.close()
	conn.close()
	es.transport.close()

def format_entity_linking(row_df):
	links_dict = {}
	links_dict['links'] = {}
	links_dict['entities'] = []
	inner_counter = 0
	for i in range(len(row_df['linking_train'][1]['links'])):
		if inner_counter >= len(row_df['entity_train'][1]):
			break
		link_item = row_df['linking_train'][1]['links'][i]
		entity_item = row_df['entity_train'][1][inner_counter]
		
		if link_item['tuple'][0] == entity_item[0] and link_item['tuple'][1] == entity_item[1]:
			links_dict['links'][tuple(link_item['tuple'])] = {link_item['entity'] : 1.0}
			links_dict['entities'].append(tuple(entity_item))
			inner_counter += 1
	if len(links_dict['links']) == 0:
		raise ValueError
	return (row_df['linking_train'][0], links_dict)


def get_custom_candidates(kb, span):
	terms_list = elastic_kb.get_training_candidate_terms(es, span.text)

	candidate_list = []
	entities = []
	for term in terms_list:
		new_candidate_list = kb.get_alias_candidates(term)
		for c in new_candidate_list:
			if c.entity_ not in entities:
				entities.append(c.entity_)
				candidate_list.append(c)
		# candidate_list.extend(kb.get_alias_candidates(term))
	# print(terms_list, entities, span.text)
	return candidate_list

def create_kb(vocab):
	kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=768)
	kb.from_disk("/home/nkaramooz/Documents/alimbic/kb")
	return kb

@spacy.registry.misc("spacy.CustomCandidateGenerator")
def create_candidates() -> Callable[[KnowledgeBase, "Span"], Iterable[Candidate]]:
	return get_custom_candidates


# def get_lowercased_candidates(kb, span):
# 	print("span", span)
# 	return kb.get_alias_candidates(span.text.lower())

# @spacy.registry.misc("spacy.LowercaseCandidateGenerator.v1")
# def create_candidates() -> Callable[[KnowledgeBase, "Span"], Iterable[Candidate]]:
# 	print("DASHJKASHKDJ")
	# return get_lowercased_candidates


if __name__ == "__main__":
	print("main")

	nlp = spacy.load("/home/nkaramooz/Documents/alimbic/web_lg_output/model-best")
	# use 768 for transformer model
	# kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=300)
	# load_kb_primary_entities(kb,nlp)
	# kb.to_disk("/home/nkaramooz/Documents/alimbic/kb")
	
	kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=300)
	kb.from_disk("/home/nkaramooz/Documents/alimbic/kb")

	# load_kb_aliases(kb)
	# kb.to_disk("/home/nkaramooz/Documents/alimbic/kb")

	# kb = load_kb("/home/nkaramooz/Documents/alimbic/kb")
	# gen_linking_data_top()


	
	# terms_list = elastic_kb.get_training_candidate_terms(es, "neutropenia")
	# candidate_list = []
	# for term in terms_list:
	# 	candidate_list.extend(kb.get_alias_candidates(term))
	# for c in candidate_list:
	# 	print(c.entity_)

	term = "GBM"
	terms = list(set(elastic_kb.get_training_candidate_terms(es, term)))

	# candidates = []
	# for item in terms:
	# 	print(item)
	# 	candidates.extend(elastic_kb.get_exact_candidate_list(es,item))
		# candidates.extend(kb.get_alias_candidates(item))
	# candidates = list(set(candidates))
	# print(candidates)
	# for c in candidates:
	# 	print(c.entity_)
		# print(kb.get_alias_candidates(item))
	# print(kb.get_alias_candidates(term))
	# print([c.entity for c in kb.get_alias_candidates('ankyalosing spondylitis')])
	
	# nlp = spacy.load("/home/nkaramooz/Documents/alimbic/web_lg_output/model-best")
	# conn,cursor = pg.return_postgres_cursor()
	# query = """
	# 	select sentence_id, linking_train, entity_train, ver 
	# 	from spacy.entity_linking_train
	# 	limit 10000
	# 	"""
	# row_df = pg.return_df_from_query(cursor, query, None, ['sentence_id', 'linking_train', 'entity_train', 'ver'])

	# query = """
	# 	select sentence_id, linking_train, entity_train, ver 
	# 	from spacy.entity_linking_train
	# 	where linking_train -> 1 -> 'links' -> 0 ->> 'entity'::text = '299071' limit 50
	# """
	# row_df = row_df.append(pg.return_df_from_query(cursor, query, None, ['sentence_id', 'linking_train', 'entity_train', 'ver']), ignore_index=True)

	# query = """
	# 	select sentence_id, linking_train, entity_train, ver 
	# 	from spacy.entity_linking_train
	# 	where linking_train -> 1 -> 'links' -> 0 ->> 'entity'::text = '45674' limit 50
	# """

	# row_df = row_df.append(pg.return_df_from_query(cursor, query, None, ['sentence_id', 'linking_train', 'entity_train', 'ver']), ignore_index=True)

	# query = """
	# 	select sentence_id, linking_train, entity_train, ver 
	# 	from spacy.entity_linking_train
	# 	where linking_train -> 1 -> 'links' -> 0 ->> 'entity'::text = '5350' limit 50
	# """

	# row_df = row_df.append(pg.return_df_from_query(cursor, query, None, ['sentence_id', 'linking_train', 'entity_train', 'ver']), ignore_index=True)

	# train_dataset = row_df.apply(format_entity_linking, axis=1)

	# cursor.close()
	# conn.close()
	

	# if "sentencizer" not in nlp.pipe_names:
	# 	nlp.add_pipe("sentencizer")
	# sentencizer = nlp.get_pipe("sentencizer")
	# TRAIN_EXAMPLES = []

	# for text, annotation in train_dataset:
	# 	example = Example.from_dict(nlp.make_doc(text), annotation)
	# 	example.reference = sentencizer(example.reference)
	# 	TRAIN_EXAMPLES.append(example)

	# entity_linker = nlp.add_pipe("entity_linker", config={"incl_prior" : False, "get_candidates" : {'@misc' : 'spacy.CustomCandidateGenerator'}}, last=True)
	# entity_linker.initialize(get_examples=lambda: TRAIN_EXAMPLES, kb_loader=load_kb("/home/nkaramooz/Documents/alimbic/kb"))

	# with nlp.select_pipes(enable=["entity_linker"]):
	# 	optimizer = nlp.resume_training()
	# 	for itn in range(1000):
	# 		random.shuffle(TRAIN_EXAMPLES)
	# 		batches = minibatch(TRAIN_EXAMPLES, size=compounding(4.0, 32.0, 1.001))
	# 		losses = {}
	# 		for batch in batches:
	# 			nlp.update(batch, drop=0.2, losses=losses, sgd=optimizer,)
	# 		if itn % 50 == 0:
	# 			print(itn, "Losses", losses)
	# print(itn, "Losses", losses)
	# nlp.to_disk("/home/nkaramooz/Documents/alimbic/alimbic_el")
	# es.transport.close()

	

	# nlp = spacy.load("/home/nkaramooz/Documents/alimbic/alimbic_el")
	
	# text = "HFrEF among patients with migraines and GBM on sacubitril-valsartan."
	# doc = nlp(text)
	# for ent in doc.ents:
	# 	print(ent.text, ent.label_, ent.kb_id_)



	# nlp.replace_pipe("entity_linker", "entity_linker", config={"get_candidates": {"@misc" :"spacy.CustomCandidateGenerator"}})
	# kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=768)
	# kb.from_disk("/home/nkaramooz/Documents/alimbic/kb")
	# print("get_alias)")
	# print([c.entity_ for c in kb.get_candidates('ankyalosing spondylitis')])
	# all_entities = kb.get_alias_strings()
	# print(fuzz.ratio("Hart failure", "Heart failure"))
	# cands = [item for item in kb.get_alias_strings() if fuzz.ratio(item, "CHF") > 85]
	# print(cands)
	# text = "CHF and Hearts disease among patients."
	# doc = nlp(text)
	# for ent in doc.ents:
	# 	print(ent.text, ent.label_, ent.kb_id_)
	# 	print(text)
	# kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=768)
	# kb.from_disk("/home/nkaramooz/Documents/alimbic/kb")
	# print(f"Heart disease': {[c.entity_ for c in kb.get_alias_candidates('Heart disease')]}")