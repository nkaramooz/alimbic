import xml.etree.ElementTree as ET 
import json
from nltk.stem.wordnet import WordNetLemmatizer
import sys 
import codecs
import snomed_annotator as ann
import nltk.data
import pandas as pd
import multiprocessing as mp
import utilities.utils as u, utilities.pglib as pg
import os
import datetime
from multiprocessing import Pool
import copy
import boto3
import re


INDEX_NAME = 'pubmedx1'

def doc_worker(input):
	for func,args in iter(input.get, 'STOP'):
		doc_calculate(func, args)


def doc_calculate(func, args):
	func(*args)


def index_doc_from_elem(elem, filter_words_df, filename):
	if elem.tag != 'PubmedArticle':
		raise ValueError('lost element')

		# american journal of hypertension
	if (is_issn(elem, '0895-7061') or is_issn(elem, '1941-7225') \
		# hypertension
		or is_issn(elem, '0194-911X') or is_issn(elem, '1524-4563')\
		# cochrane database of systematic reviews
		or is_issn(elem, '1469-493X') or is_issn(elem, '1465-1858')\
		# british medial journal
		or is_issn(elem, '0959-8138') or is_issn(elem, '1756-1833')\
		# Lung
		or is_issn(elem, '0341-2040') or is_issn(elem, '1432-1750')\
		# Circulation. Heart failure
		or is_issn(elem, '1941-3289') or is_issn(elem, '1941-3297')\
		# NEJM
		or is_issn(elem, '1533-4406') or is_issn(elem, '0028-4793')\
		# American family physician
		or is_issn(elem, '0002-838X') or is_issn(elem, '1532-0650')\
		# Annals of internal medicine
		or is_issn(elem, '0003-4819') or is_issn(elem, '1539-3704')\
		# JAMA
		or is_issn(elem, '0098-7484') or is_issn(elem, '1538-3598')\
		# Annals of american thoracic society
		or is_issn(elem, '2325-6621') or is_issn(elem, '1943-5665')\
		# Lancet
		or is_issn(elem, '0140-6736') or is_issn(elem, '1474-547X')\
		# Circulation
		or is_issn(elem, '0009-7322') or is_issn(elem, '1524-4539')):

		json_str = {}
		json_str = get_journal_info(elem, json_str)
		if json_str['journal_pub_year'] is not None:
			if (int(json_str['journal_pub_year']) > 1990):
	
				json_str = get_article_info(elem, json_str)
				
				if (not bool(set(json_str['article_type']) & set(['Letter', 'Editorial', 'Comment', 'Biography', 'Patient Education Handout', 'News']))):

					json_str = get_pmid(elem, json_str)

					json_str = get_article_ids(elem, json_str)
					
					json_str['citations_pmid'] = get_article_citations(elem)


					title_annotation = get_snomed_annotation(json_str['article_title'], filter_words_df)
					json_str['title_conceptids'] = title_annotation['conceptid'].tolist()
					json_str['title_dids'] = title_annotation['description_id'].tolist()

					json_str['abstract_conceptids'], json_str['abstract_dids'] = get_abstract_conceptids(json_str['article_abstract'], filter_words_df)
	
					json_str['index_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
		
					json_str['index_time'] = datetime.datetime.now().strftime("%H:%M:%S")
	
					json_str['filename'] = filename
					pmid = json_str['pmid']
					json_str =json.dumps(json_str)
					json_obj = json.loads(json_str)

					es = u.get_es_client()
					get_article_query = {'_source': ['id', 'pmid'], 'query': {'constant_score': {'filter' : {'term' : {'pmid': pmid}}}}}
					query_result = es.search(index=INDEX_NAME, body=get_article_query)
						
					if query_result['hits']['total'] == 0 or query_result['hits']['total'] > 1:
						try:
							es.index(index=INDEX_NAME, doc_type='abstract', body=json_obj)
						except:
							u.pprint(json_obj)
							u.pprint(json_str)
							raise ValueError('incompatible json obj')
						
						
					elif query_result['hits']['total'] == 1:
						article_id = query_result['hits']['hits'][0]['_id']
						es.index(index=INDEX_NAME, id=article_id, doc_type='abstract', body=json_obj)
	elem.clear()




def load_local(start_file, filter_words_df):
	es = u.get_es_client()
	number_of_processes = 60
	pool = Pool(processes=number_of_processes)

	index_exists = es.indices.exists(index=INDEX_NAME)
	if not index_exists:
		es.indices.create(index=INDEX_NAME, body={})

	folder_arr = ['resources/production_baseline_2']

	for folder_path in folder_arr:
		file_counter = 0

		for filename in os.listdir(folder_path):
			abstract_counter = 0
			file_path = folder_path + '/' + filename
			
			file_num = int(re.findall('medline17n(.*).xml', filename)[0])

			if file_num >= start_file:

				print(filename)
			
				file_timer = u.Timer('file')

				tree = ET.parse(file_path)		
				root = tree.getroot()

				file_abstract_counter = 0

				for elem in root:
					if elem.tag == 'PubmedArticle':
						params = (elem, filter_words_df, filename)
						pool.apply_async(index_doc_from_elem, params)
						file_abstract_counter += 1
						abstract_counter += 1

					elif elem.tag == 'DeleteCitation':
						delete_pmid_arr = get_deleted_pmid(elem)

						for pmid in delete_pmid_arr:
							get_article_query = {'_source': ['id', 'pmid'], 'query': {'constant_score': {'filter' : {'term' : {'pmid': pmid}}}}}
							query_result = es.search(index=INDEX_NAME, body=get_article_query)

							if query_result['hits']['total'] == 0:
								continue
							elif query_result['hits']['total'] == 1:
								article_id = query_result['hits']['hits'][0]['_id']
								es.delete(index=INDEX_NAME, doc_type='abstract', id=article_id)
							else:
								print("delete: more than one document found")
								print(pmid)
						elem.clear()
					else:
						elem.clear()
		
				file_timer.stop()
				
				if file_num >= start_file+10:
					break

		# if file_num+10 < 893:
		# 	break
	pool.close()
	pool.join()


def load_pubmed_local_2(start_file, filter_words_df):
	es = u.get_es_client()
	number_of_processes = 8
	
	task_queue = mp.Queue()

	pool = []

	for i in range(number_of_processes):
		p = mp.Process(target=doc_worker, args=(task_queue,))
		pool.append(p)
		p.start()


	index_exists = es.indices.exists(index=INDEX_NAME)
	if not index_exists:
		es.indices.create(index=INDEX_NAME, body={})

	folder_arr = ['resources/production_baseline_2', 'resources/production_updates_10_21_17']

	for folder_path in folder_arr:
		file_counter = 0

		for filename in os.listdir(folder_path):
			abstract_counter = 0
			file_path = folder_path + '/' + filename
			
			file_num = int(re.findall('medline17n(.*).xml', filename)[0])

			if file_num >= start_file:


				print(filename)
			
				file_timer = u.Timer('file')

				tree = ET.parse(file_path)		
				root = tree.getroot()

				file_abstract_counter = 0

				for elem in root:
					if elem.tag == 'PubmedArticle':
						params = (elem, filter_words_df, filename)
						task_queue.put((index_doc_from_elem, params))
						file_abstract_counter += 1
						abstract_counter += 1

					elif elem.tag == 'DeleteCitation':
						delete_pmid_arr = get_deleted_pmid(elem)

						for pmid in delete_pmid_arr:
							get_article_query = {'_source': ['id', 'pmid'], 'query': {'constant_score': {'filter' : {'term' : {'pmid': pmid}}}}}
							query_result = es.search(index=INDEX_NAME, body=get_article_query)

							if query_result['hits']['total'] == 0:
								continue
							elif query_result['hits']['total'] == 1:
								article_id = query_result['hits']['hits'][0]['_id']
								es.delete(index=INDEX_NAME, doc_type='abstract', id=article_id)
							else:
								print("delete: more than one document found")
								print(pmid)
						elem.clear()
					else:
						elem.clear()

				file_timer.stop()
				if file_num >= start_file+10:
					break
				
		if file_num+10 < 893:
			break
				
				

	for i in range(number_of_processes):
		task_queue.put('STOP')

	for p in pool:
		p.join()

	

def aws_load_pubmed():
	es = u.get_es_client()
	number_of_processes = 16
	task_queue = mp.Queue()
	pool = []

	for i in range(number_of_processes):
		p = mp.Process(target=doc_worker, args=(task_queue,))
		pool.append(p)
		p.start()

	cursor = pg.return_postgres_cursor()
	filter_words_query = "select words from annotation.filter_words"
	filter_words_df = pg.return_df_from_query(cursor, filter_words_query, None, ["words"])
	cursor.close()

	index_exists = es.indices.exists(index=INDEX_NAME)
	if not index_exists:
		es.indices.create(index=INDEX_NAME, body={})

	s3 = boto3.resource('s3')
	bucket = s3.Bucket('pubmed-baseline-1')
	abstract_counter = 0

	for object in bucket.objects.all():
		
		file_num = int(re.findall('medline17n(.*).xml', object.key)[0])

		if file_num >= 600:

			bucket.download_file(object.key, object.key)
			print(object.key)
		
			file_timer = u.Timer('file')

			tree = ET.parse(object.key)		
			root = tree.getroot()

			file_abstract_counter = 0

			for elem in root:
				if elem.tag == 'PubmedArticle':
					params = (elem, filter_words_df, object.key)
					task_queue.put((index_doc_from_elem, params))
					file_abstract_counter += 1

				elif elem.tag == 'DeleteCitation':
					delete_pmid_arr = get_deleted_pmid(elem)

					for pmid in delete_pmid_arr:
						get_article_query = {'_source': ['id', 'pmid'], 'query': {'constant_score': {'filter' : {'term' : {'pmid': pmid}}}}}
						query_result = es.search(index=INDEX_NAME, body=get_article_query)

						if query_result['hits']['total'] == 0:
							continue
						elif query_result['hits']['total'] == 1:
							article_id = query_result['hits']['hits'][0]['_id']
							es.delete(index=INDEX_NAME, doc_type='abstract', id=article_id)
						else:
							print("delete: more than one document found")
							print(pmid)
					elem.clear()
				else:
					elem.clear()

			os.remove(object.key)

			file_timer.stop()

	for i in range(number_of_processes):
		task_queue.put('STOP')

	for p in pool:
		p.join()


def aws_load_pubmed_2(start_file, filter_words_df):
	es = u.get_es_client()
	number_of_processes = 60
	pool = Pool(processes=number_of_processes)

	index_exists = es.indices.exists(index=INDEX_NAME)
	if not index_exists:
		es.indices.create(index=INDEX_NAME, body={})

	s3 = boto3.resource('s3')
	bucket = s3.Bucket('pubmed-baseline-1')
	abstract_counter = 0

	for object in bucket.objects.all():

		file_counter = 0
		abstract_counter = 0
		
		file_num = int(re.findall('medline17n(.*).xml', object.key)[0])

		if file_num >= start_file:
			bucket.download_file(object.key, object.key)
			print(object.key)
		
			file_timer = u.Timer('file')
			tree = ET.parse(object.key)		
			root = tree.getroot()
			file_abstract_counter = 0
			for elem in root:
				if elem.tag == 'PubmedArticle':
					params = (elem, filter_words_df, object.key)
					pool.apply_async(index_doc_from_elem, params)
					file_abstract_counter += 1
					abstract_counter += 1
				elif elem.tag == 'DeleteCitation':
					delete_pmid_arr = get_deleted_pmid(elem)
					for pmid in delete_pmid_arr:
						get_article_query = {'_source': ['id', 'pmid'], 'query': {'constant_score': {'filter' : {'term' : {'pmid': pmid}}}}}
						query_result = es.search(index=INDEX_NAME, body=get_article_query)
						if query_result['hits']['total'] == 0:
							continue
						elif query_result['hits']['total'] == 1:
							article_id = query_result['hits']['hits'][0]['_id']
							es.delete(index=INDEX_NAME, doc_type='abstract', id=article_id)
						else:
							print("delete: more than one document found")
							print(pmid)
					elem.clear()
				else:
					elem.clear()
			os.remove(object.key)
			file_timer.stop()
			
		if file_num >= start_file+10:
			break

	pool.close()
	pool.join()


def get_abstract_conceptids(abstract_dict, filter_words_df):
	cid_dict = {}
	did_dict = {}

	if abstract_dict is not None:
		for k1 in abstract_dict:
			sub_cid_dict = {}
			sub_did_dict = {}
			for k2 in abstract_dict[k1]:
				res = get_snomed_annotation(abstract_dict[k1][k2], filter_words_df)
				sub_cid_dict[k2] = res['conceptid'].tolist()
				sub_did_dict[k2] = res['description_id'].tolist()
			cid_dict[k1] = sub_cid_dict
			did_dict[k1] = sub_did_dict

		return cid_dict, did_dict
	else:
		return None

def get_deleted_pmid(elem):
	delete_pmid_arr = []
	for item in elem:
		delete_pmid_arr.append(item.text)

	return delete_pmid_arr

def get_article_citations(elem):
	citation_elems = elem.find('*/CommentsCorrectionsList')
	citation_pmid_list = []
	if citation_elems is not None:
		for citation in citation_elems:
			for item in citation:
				if citation.attrib['RefType'] == 'Cites':
					for item in citation:
						if item.tag == 'PMID':
							citation_pmid_list.append(item.text)
		return citation_pmid_list
	else:
		return None

def get_article_ids(elem, json_str):
	id_list_elem = elem.find('*/ArticleIdList')
	article_id_dict = {}
	for elem_id in id_list_elem:
		if elem_id.attrib['IdType'] == 'pubmed':
			article_id_dict['pmid'] = elem_id.text
		else:
			article_id_dict[elem_id.attrib['IdType']] = elem_id.text
	json_str['article_ids'] = article_id_dict

	return json_str

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
		json_str['journal_volume'] = None
		json_str['journal_issue'] = None
		json_str['journal_iso_abbrev'] = None
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
		iso_elem = journal_elem.find('./ISOAbbreviation')
		json_str['journal_iso_abbrev'] = iso_elem.text 
	except:
		json_str['journal_iso_abbrev'] = None

	try:
		journal_volume_elem = journal_elem.find('./JournalIssue/Volume')
		json_str['journal_volume'] = journal_volume_elem.text
	except:
		json_str['journal_volume'] = None

	try:
		journal_issue_elem = journal_elem.find('./JournalIssue/Issue')
		json_str['journal_issue'] = journal_issue_elem.text
	except:
		json_str['journal_issue'] = None

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

	try:
		article_elem = elem.find('./MedlineCitation/Article')
	except:
		json_str['article_title'] = None
		json_str['article_abstract'] = None
		json_str['article_type'] = None
		json_str['article_type_id'] = None

	try:
		title_elem = article_elem.find('./ArticleTitle')
		json_str['article_title'] = title_elem.text
	except:
		json_str['article_title'] = None

	try:
		abstract_elem = article_elem.find('./Abstract')
		abstract_dict = {}

		for abstract_sub_elem in abstract_elem:

			sub_elem_dict = {}
			
			if not abstract_sub_elem.attrib:
				sub_elem_dict['text'] = abstract_sub_elem.text
				abstract_dict['text'] =  sub_elem_dict
			else:
				if abstract_sub_elem.attrib['Label'] == "":
					sub_elem_dict["unassigned"] == abstract_sub_elem.text
				else:
					sub_elem_dict[abstract_sub_elem.attrib['Label'].lower()] = abstract_sub_elem.text 

				try:
					abstract_dict[abstract_sub_elem.attrib['NlmCategory'].lower()] = sub_elem_dict
				except:
					abstract_dict[abstract_sub_elem.attrib['Label'].lower()] = sub_elem_dict

		json_str['article_abstract'] = abstract_dict

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

def get_snomed_annotation(text, filter_words_df):
	cursor = pg.return_postgres_cursor()
	if text is None:
		return None
	else:
		annotation = ann.annotate_text_not_parallel(text, filter_words_df, cursor, True)

		if annotation is not None:
			return annotation
		else:
			return None
	cursor.close()

if __name__ == "__main__":

	cursor = pg.return_postgres_cursor()
	
	filter_words_query = "select words from annotation.filter_words"
	filter_words_df = pg.return_df_from_query(cursor, filter_words_query, None, ["words"])
	cursor.close()

	start_file = 510
	while (start_file < 893):
		print(start_file)
		load_pubmed_local_2(start_file, filter_words_df)
		start_file += 11



