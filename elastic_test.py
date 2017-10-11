import sys
import requests
import pglib as pg
from elasticsearch import Elasticsearch
import json
import pandas as pd
import codecs
import utils as u
import nltk.data
import multiprocessing as mp
import sys
import itertools

# r = requests.get('http://localhost:9200')
# print(r.content)
# es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# file_system_df = pd.read_pickle('file_system')

# for doc_index, doc in file_system_df.iterrows():
# 	timer = u.Timer("50 sentences")
# 	file_path = "/Users/LilNimster/Documents/wiki_data/text_files/"
# 	file_path += doc['filename']
# 	current_doc = codecs.open(file_path, 'r', encoding='utf8')
# 	doc_text = current_doc.read()
# 	tokenized = nltk.sent_tokenize(doc_text)

# 	if doc['filename'] == 'e3b0949d-4549-43f0-b3b3-fdce04dca74b.txt':
# 		index_timer = u.Timer("index timer")
# 		file_path = "/Users/LilNimster/Documents/wiki_data/text_files/"
# 		file_path += doc['filename']

# 		current_doc = codecs.open(file_path, 'r', encoding='utf8')

# 		for ln_index, line in enumerate(tokenized):
# 			line_json = {'ln_num' : ln_index, 'doc_id': doc['docid'], 'body': line }
# 			es.index(index= 'sw', doc_type='medical', body=line_json)
# 		index_timer.stop()

# 		break



# 		doc_text = {'body': current_doc.read()}

# 		es.index(index='pe', doc_type='medical', id=1, body=doc_text)

# print(es.get(index='pe', doc_type='medical', id=1))

# print(es.search(index="pe", body={"explain" : True, "query" : {"match": {'body' : 'cough'}}}))jps | grep Elasticsearch
# query = {"aggs": {"body": {"terms": "pe"}}}
# res_json = es.search(index='sw', body={"query" : {"match" : {'body' : 'hemoptysis'}}})

# res_count = res_json['hits']
# for value in res_count['hits']:
# 	print value['_source']['ln_num']
# print len(res_count['hits'])
# print(es.search(index='sw', body=query))
# print(es.get(index='sw', doc_type='medical', id=))

def elastic_parallel():
	cursor = pg.return_postgres_cursor()
	query = "select conceptid, term from annotation.selected_concept_descriptions"
	ann_df = pg.return_df_from_query(cursor, query, ['conceptid', 'term'])

	number_of_processes = 50

	task_q = mp.Queue()
	done_q = mp.Queue()
	results_df = pd.DataFrame()
	print(len(ann_df))
	sys.exit(0)
	print("cp1")
	for index,row in ann_df.iterrows():
		params = (row['term'], row['conceptid'])
		task_q.put((annotate_with_concept, params))

	print("cp2")
	for i in range(number_of_processes):
		mp.Process(target=worker, args=(task_queue, done_queue)).start()

	for i in range(len(task_q)):
		results_df = results_df.append(done_q.get())

	for i in range(number_of_processes):
		task_queue.put('STOP')

	engine = pg.return_sql_alchemy_engine()
	results_df.to_sql('elastic_annotation', engine, schema='annotation', if_exists='append')


def worker(input, output):

	for func, args in iter(input.get, 'STOP'):
		result = calculate(func,args)
		output.put(result)

def calculate(func, args):

	result = func(*args)
	return result

def annotate_with_concept(term, conceptid):
	results_df = pd.DataFrame()
	es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
	es_query = {"query" : {"match": {'body' : term}}}
	res_json = es.search(index='sw', body=es_query)

	if res_json['hits']['total'] > 0:

		for value in res_json['hits']['hits']:
			result = pd.DataFrame([[value['_source']['ln_num'], 
				conceptid,
				term,
				value['_source']['doc_id']
				]], columns=['ln_num', 'conceptid', 'term', 'docid'])
			results_df = results_df.append(result)
	return results_df
# engine = pg.return_sql_alchemy_engine()
# results_df.to_sql('elastic_annotation', engine, schema='annotation', if_exists='append')

# elastic_parallel()

def print_all(*args):
	for value in args:
		for sub in value:
			yield sub


def chain(*iterables):
	print iterables
	for it in iterables:
		for element in it:
			print element 
			yield element

print(list(chain([('1','2')])))

# test_df = pd.DataFrame()
# val1 = pd.DataFrame([['ABC', 'A', '2']])
# val2 = pd.DataFrame([['ABC', 'A', '3']])
# val3 = pd.DataFrame([['DEF', 'C', '1']])
# val4 = pd.DataFrame([['ABC', 'D', '9']])
# test_df = test_df.append(val1)
# test_df = test_df.append(val2)
# test_df = test_df.append(val3)
# test_df =test_df.append(val4)
# test_df.columns = ['key', 'SUB', 'value']

# group = test_df.groupby(['key', 'SUB'], as_index=False)
# new = group.aggregate(lambda x: tuple(x))
# tup = new[new['key']=='DEF']['value']
# print tup
# print list(itertools.chain(*tup))
# print(list(print_all(tup)))
# print(tup)

