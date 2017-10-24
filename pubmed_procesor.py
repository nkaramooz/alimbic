import xml.etree.ElementTree as ET 
import json
from elasticsearch import Elasticsearch
import sys 
import codecs
import snomed_annotator as snomed
import nltk.data
import pandas as pd
import pglib as pg
import multiprocessing as mp
import utils as u
import os

def load_pubmed_baseline():
	folder_path = 'resources/production_baseline_2'
	
	es = Elasticsearch([{'host' : 'localhost', 'port' : 9200}])
	
	cursor = pg.return_postgres_cursor()
	
	filter_words_query = "select words from annotation.filter_words"
	filter_words_df = pg.return_df_from_query(cursor, filter_words_query, None, ["words"])
	
	for filename in os.listdir(folder_path):
		file_timer = u.Timer('file')
		file_path = folder_path + '/' + filename

		tree = ET.parse(file_path)

		root = tree.getroot()

		file_abstract_counter = 0
		for elem in root:
			# Filtering only for NEJM articles
			if (is_issn(elem, '1533-4406') or is_issn(elem, '0028-4793')):

				json_str = {}
				json_str = get_pmid(elem, json_str)
				json_str = get_journal_info(elem, json_str)
				json_str = get_article_info(elem, json_str)
				json_str['title_conceptids'] = get_snomed_annotation(json_str['article_title'], filter_words_df)
				json_str['abstract_conceptids'] = get_snomed_annotation(json_str['article_abstract'], filter_words_df)
				json_str =json.dumps(json_str)
				json_obj = json.loads(json_str)

				es.index(index='pubmed', doc_type='abstract', body=json_obj)
				file_abstract_counter += 1
		print(file_abstract_counter)
		file_timer.stop()

def load_pubmed_updates():
	folder_path = 'resources/production_baseline_2'

	es = Elasticsearch([{'host' : 'localhost', 'port' : 9200}])
	
	cursor = pg.return_postgres_cursor()
	
	filter_words_query = "select words from annotation.filter_words"
	filter_words_df = pg.return_df_from_query(cursor, filter_words_query, None, ["words"])

	for filename in os.listdir(folder_path):
		file_timer = u.Timer('file')
		file_path = folder_path + '/' + filename

		tree = ET.parse(file_path)

		root = tree.getroot()

		file_abstract_counter = 0
		for elem in root:
			if elem.tag == 'PubmedArticle':
				# Filtering only for NEJM articles
				if (is_issn(elem, '1533-4406') or is_issn(elem, '0028-4793')):

					json_str = {}
					json_str = get_pmid(elem, json_str)
					json_str = get_journal_info(elem, json_str)
					json_str = get_article_info(elem, json_str)
					json_str['title_conceptids'] = get_snomed_annotation(json_str['article_title'], filter_words_df)
					json_str['abstract_conceptids'] = get_snomed_annotation(json_str['article_abstract'], filter_words_df)
					pmid = json_str['pmid']
					json_str =json.dumps(json_str)
					json_obj = json.loads(json_str)
					# need to check if abstract already exists
					

					get_article_query = {'_source': ['id', 'pmid'], 'query': {'constant_score': {'filter' : {'term' : {'pmid': pmid}}}}}
					query_result = es.search(index='pubmed', body=get_article_query)

					if query_result['hits']['total'] == 0:
						es.index(index='pubmed', doc_type='abstract', body=json_obj)
					elif query_result['hits']['total'] == 1:
						article_id = query_result['hits']['hits'][0]['_id']
						es.index(index='pubmed', id=article_id, doc_type='abstract', body=json_obj)
					else:
						print("###more than one result found###")
					file_abstract_counter += 1
			elif elem.tag == 'DeleteCitation':

				delete_pmid_arr = get_deleted_pmid(elem)

				for pmid in delete_pmid_arr:
					get_article_query = {'_source': ['id', 'pmid'], 'query': {'constant_score': {'filter' : {'term' : {'pmid': pmid}}}}}
					query_result = es.search(index='pubmed', body=get_article_query)

					if query_result['hits']['total'] == 0:
						continue
					elif query_result['hits']['total'] == 1:
						article_id = query_result['hits']['hits'][0]['_id']
						es.delete(index='pubmed', doc_type='abstract', id=article_id)
					else:
						print("delete: more than one document found")
						print(pmid)
			
		print(file_abstract_counter)
		file_timer.stop()

def get_deleted_pmid(elem):
	delete_pmid_arr = []
	for item in elem:
		delete_pmid_arr.append(item.text)

	return delete_pmid_arr

def is_issn(elem, issn):
	try:
		j_list = elem.findall('./MedlineCitation/Article/Journal')
		journal_elem = j_list[0]
		issn_elem = journal_elem.findall('./ISSN')[0]
		issn_text = issn_elem.text

		if issn_text == issn:
			return True
		else:
			return False
	except:
		return False

def get_pmid(elem, json_str):
	pmid = elem.findall('*/PMID')
	try:
		json_str['pmid'] = pmid[0].text
	except:
		json_str['pmid'] = None
	return json_str

def get_journal_info(elem, json_str):
	j_list = elem.findall('./MedlineCitation/Article/Journal')

	try:
		journal_elem = j_list[0]
	except:
		json_str['journal_title'] = None
		json_str['journal_pub_year'] = None
		json_str['journal_pub_month'] = None
		json_str['journal_pub_day'] = None
		json_str['journal_issn'] = None
		return json_str
	
	try:
		issn_elem = journal_elem.findall('./ISSN')[0]
		json_str['journal_issn'] = issn_elem.text
		json_str['journal_issn_type'] = issn_elem.attrib
	except:
		json_str['journal_issn'] = None
		json_str['journal_issn_type'] = None

	try:
		title_elem = journal_elem.findall('./Title')[0]
		json_str['journal_title'] = title_elem.text
	except:
		json_str['journal_title'] = None

	try:
		year_elem = journal_elem.findall('./JournalIssue/PubDate/Year')[0]
		json_str['journal_pub_year'] = year_elem.text
	except:
		json_str['journal_pub_year'] = None

	try:
		month_elem = journal_elem.findall('./JournalIssue/PubDate/Month')[0]
		json_str['journal_pub_month'] = month_elem.text
	except:
		json_str['journal_pub_month'] = None

	try:
		day_elem = journal_elem.findall('./JournalIssue/PubDate/Day')[0]
		day_str['journal_pub_month'] = day_elem.text
	except:
		json_str['journal_pub_day'] = None

	return json_str

def get_article_info(elem, json_str):
	art_list = elem.findall('./MedlineCitation/Article')

	try:
		article_elem = art_list[0]
	except:
		json_str['article_title'] = None
		json_str['article_abstract'] = None
		json_str['article_type'] = None
		json_str['article_type_id'] = None

	try:
		title_elem = article_elem.findall('./ArticleTitle')[0]
		json_str['article_title'] = title_elem.text
	except:
		json_str['article_title'] = None

	try:
		#also probably can just do a for loop
		#if label is null then do 'article_abstract'
		#else maybe article_abstract_LABEL#
		#store NLM category as well
		abstract_elem = article_elem.findall('./Abstract/AbstractText')[0] ## if label is null then only one, otherwise more
		json_str['article_abstract'] = abstract_elem.text
	except:
		json_str['article_abstract'] = None

	try:
		article_type_elem = article_elem.findall('./PublicationTypeList/PublicationType')
		json_str['article_type'] = []
		json_str['article_type_id'] = []

		for node in article_type_elem:
			json_str['article_type'].append(node.text)
			json_str['article_type_id'].append(node.attrib['UI'])
	except:
		json_str['article_type'] = None
		json_str['article_type_id'] = None

	return json_str



#This is deprecated but may be useful in the future
def get_json(elem):
	if elem.text is None:
		return None
	else:
		new_result = {}

		if elem.text.strip("\n").strip(" ") != '':
			return elem.text
		else:
			for node in elem:
				if node.tag not in new_result:
					new_result[node.tag] = get_json(node)
				elif type(new_result[node.tag]) != list:
					old_dict = new_result[node.tag]
					new_result[node.tag] = [old_dict]
					new_result[node.tag].append(get_json(node))
				else:
					new_result[node.tag].append(get_json(node))
		return new_result

def get_snomed_annotation(text, filter_words_df):
	
	if text is None:
		return None
	else:
		annotation = annotate_text(text, filter_words_df)

		if annotation is not None:
			return annotation['conceptid'].tolist()
		else:
			return None

def annotate_text(text, filter_words_df):
	number_of_processes = 60
	tokenized = nltk.sent_tokenize(text)
	results = pd.DataFrame()
	funclist = []
	task_list = []
	results_df = pd.DataFrame()

	for ln_index, line in enumerate(tokenized):
		params = (line, filter_words_df)
		task_list.append((annotate_line, params))

	task_queue = mp.Queue()
	done_queue = mp.Queue()

	for task in task_list:
		task_queue.put(task)

	for i in range(number_of_processes):
		mp.Process(target=worker, args=(task_queue, done_queue)).start()

	for i in range(len(task_list)):
		results_df = results_df.append(done_queue.get())

	for i in range(number_of_processes):
		task_queue.put('STOP')

	if len(results_df) > 0:
		return results_df
	else:
		return None

def worker(input, output):
	for func, args in iter(input.get, 'STOP'):
		result = calculate(func, args)
		output.put(result)

def calculate(func, args):
	return func(*args)

def annotate_line(line, filter_words_df):
	cursor = pg.return_postgres_cursor()
	# line = line.encode('utf-8')
	line = line.replace('.', '')
	line = line.replace('!', '')
	line = line.replace(',', '')
	line = line.replace(';', '')
	line = line.replace('*', '')
	line = line.replace('[', '')
	line = line.replace(']', '')
	line = line.replace('-', '')
	annotation = snomed.return_line_snomed_annotation(cursor, line, 100, filter_words_df)

	return annotation


######### MIGRATIONS

def get_article_type(elem):
	art_list = elem.findall('./MedlineCitation/Article')
	article_type_dict = {}

	try:
		article_elem = art_list[0]
	except:
		article_type_dict['article_type'] = None
		article_type_dict['article_type_id'] = None

	try:
		article_type_elem = article_elem.findall('./PublicationTypeList/PublicationType')
		article_type_dict['article_type'] = []
		article_type_dict['article_type_id'] = []

		for node in article_type_elem:
			article_type_dict['article_type'].append(node.text)
			article_type_dict['article_type_id'].append(node.attrib['UI'])
	except:
		article_type_dict['article_type'] = None
		article_type_dict['article_type_id'] = None

	return article_type_dict

def add_article_types():
	folder_path = 'resources/production_baseline_1'
	es = Elasticsearch([{'host' : 'localhost', 'port' : 9200}])

	for filename in os.listdir(folder_path):
		file_timer = u.Timer('file')
		file_path = folder_path + '/' + filename
		tree = ET.parse(file_path)

		root = tree.getroot()

		file_abstract_counter = 0
		for elem in root:
			# Filtering only for NEJM articles
			if is_issn(elem, '0028-4793'):

				json_str = {}
				pmid = get_pmid(elem, {})['pmid']
				json_str['doc'] = get_article_type(elem)

				qa
				article_id = es.search(index='pubmed', body=get_article_query)['hits']['hits'][0]['_id']

				json_str =json.dumps(json_str)
				json_obj = json.loads(json_str)
				
				es.update(index='pubmed', id=article_id, doc_type='abstract', body=json_obj)


				file_abstract_counter += 1
		print(file_abstract_counter)
		file_timer.stop()


if __name__ == "__main__":
	t = u.Timer("full")
	# add_article_types()
	load_pubmed_updates()
	t.stop()

