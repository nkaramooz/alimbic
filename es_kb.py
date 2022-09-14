import psycopg2
import json
import utilities.utils2 as u, utilities.pglib as pg
import sqlalchemy as sqla
from tqdm import tqdm
from snomed_annotator2 import clean_text

INDEX_NAME = 'kb'

def create_es_index():
	es = u.get_es_client()
	index_exists = es.indices.exists(index=INDEX_NAME)
	if not index_exists:
		settings = {"mappings" :{"_doc" : {"properties" : {
			"a_cid" : {"type" : "keyword"}
			,"name" : {"type" : "text"}
			,"name_keyword" : {"type" : "keyword"}
		}}}}	
		es.indices.create(index=INDEX_NAME, body=settings, include_type_name=True)

def load_entities():
	conn,cursor = pg.return_postgres_cursor()
	es = u.get_es_client()
	query = """
		select distinct(adid) 
		from annotation2.lemmas
		where acid in (
			select root_acid from annotation2.concept_types
			where rel_type in ('condition', 'chemical', 'treatment', 'outcome', 'statistic', 'symptom', 'diagnostic', 'cause', 'study_design'
			) and (active=1 or active=3) )
	"""
	a_dids = list(pg.return_df_from_query(cursor, query, None, ['a_did'])['a_did'])
	
	for a_did in tqdm(a_dids):
		query = "select distinct(term), acid from annotation2.lemmas where adid=%s"
		term_df = pg.return_df_from_query(cursor, query, (a_did,), ['term','a_cid'])
		term = term_df['term'][0]
		term = clean_text(term)
		a_cid = term_df['a_cid'][0]
		json_str = {}
		json_str['a_cid'] = a_cid
		json_str['a_did'] = a_did
		json_str['name'] = term
		json_str['name_keyword'] = term.lower()
		json_str = json.dumps(json_str)
		json_obj = json.loads(json_str)
		es.index(index=INDEX_NAME, body=json_obj)

	cursor.close()
	conn.close()
	es.transport.close()

# Query lowered because terms are indexed on lowercase for text field in elastic search

def get_fuzzy_query(words_list):
	query = {}
	query["query"] = {}
	query["query"]["bool"] = {"should" : []}
	for word in words_list:
		query["query"]["bool"]["should"].append({"fuzzy" : {"name" : word}})
		query["query"]["bool"]["should"].append({"fuzzy" : {"name_keyword" : word}})
	return query

def get_exact_query(text):
	query = {"query" : {"term" : {"name_keyword" : text.lower()}}}
	return query

def get_candidate_list(es, text):
	text = clean_text(text.lower()).replace('+', ' ')
	words_list = text.split()
	es_query = get_fuzzy_query(words_list)
	sr = es.search(index=INDEX_NAME, body=es_query)
	a_cid_list = []
	counter = 0
	max_score = 0
	for hit in sr['hits']['hits']:
		if counter == 0:
			max_score = hit['_score']

		if hit['_score'] < 0.7*max_score:
			break
		else:  
			a_cid = hit['_source']['a_cid']
			if a_cid not in a_cid_list:
				a_cid_list.append(a_cid)
				counter += 1

			if counter == 4:
				break
	return a_cid_list

def get_exact_candidate_list(es, text):
	es_query = get_exact_query(text)
	sr = es.search(index=INDEX_NAME, body=es_query)
	a_cid_list = []
	counter = 0
	for hit in sr['hits']['hits']:
		a_cid = hit['_source']['a_cid']
		if a_cid not in a_cid_list:
			a_cid_list.append(a_cid)
			counter += 1

		if counter == 4:
			break
	return a_cid_list


def get_training_candidate_terms(es, text):
	text = clean_text(text.lower()).replace('+', ' ')
	words_list = text.split()

	es_query = get_fuzzy_query(words_list)
	sr = es.search(index=INDEX_NAME, body=es_query)

	a_cid_list = []
	terms_list = []
	counter = 0
	max_score = 0
	for hit in sr['hits']['hits']:
		if counter == 0:
			max_score = hit['_score']

		if hit['_score'] < 0.7*max_score:
			break
		else:
			a_cid = hit['_source']['a_cid']
			term = hit['_source']['name']
			if a_cid not in a_cid_list and term not in terms_list:
				a_cid_list.append(a_cid)
				terms_list.append(term)
				counter += 1

			if counter == 4:
				break

	return terms_list

if __name__=='__main__':
	print("main")
	# create_es_index()
	# load_entities()
	es = u.get_es_client()
	term = "10 day exam abnormal  for observation"
	query = {"query" : {"term" : {"name_keyword" : term.lower()}}}
	sr = es.search(index=INDEX_NAME, body=query)
	# print(sr)
	# print(get_exact_query(term))
	# print(get_exact_candidate_list(es, "Diastolic heart failure stage D"))
	print(get_candidate_list(es, "Heart attack"))
	print(get_training_candidate_terms(es, "Heart attack"))
	es.transport.close()