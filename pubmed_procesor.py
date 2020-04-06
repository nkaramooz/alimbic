from lxml import etree as ET
import json
import codecs
import snomed_annotator as ann
import pandas as pd
import multiprocessing as mp
import utilities.utils as u, utilities.pglib as pg
import os
import datetime
from multiprocessing import Pool
import copy
import re
import io
from nltk.stem.wordnet import WordNetLemmatizer
import nltk.data
import sys

INDEX_NAME = 'pubmedx1.5'

def doc_worker(input, conn,cursor):
	for func,args in iter(input.get, 'STOP'):
		doc_calculate(func, args, conn,cursor)


def doc_calculate(func, args, conn, cursor):
	func(*args, conn, cursor)


def index_doc_from_elem(elem, filename, issn_list, conn, cursor):

	elem = ET.parse(io.BytesIO(elem))
	# if elem.tag != 'PubmedArticle':
	# 	raise ValueError('lost element')

		
	issn = return_issn(elem)
	# american journal of hypertension
	# hypertension
	# cochrane database of systematic reviews
	# british medial journal
	# Lung
	# Circulation. Heart failure
	# NEJM
	# American family physician
	# Annals of internal medicine
	# JAMA
	# Annals of american thoracic society
	# Lancet
	# Neurlogy
	# Circulation
	# Pulmonology
	# Gastroenterology
	# Annals of Emergency Medicine
	# Annals of internal medicine
	# American journal of respiratory and critical care medicine
	# Journal of urology
	# Gold journal
	# Add Blue
	# Journal of hepatology
	# JACC
	# JACC Heart failure
	if issn in issn_list:
	
		json_str = {}
		json_str = get_journal_info(elem, json_str)

		if json_str['journal_pub_year'] is not None:
			if (int(json_str['journal_pub_year']) >= 2020):
				json_str, article_text = get_article_info_2(elem, json_str)
		
				if (not bool(set(json_str['article_type']) & set(['Letter', 'Editorial', 'Comment', 'Biography', 'Patient Education Handout', 'News']))):
					# conn, cursor = pg.return_postgres_cursor()
					
					json_str = get_pmid(elem, json_str)
					json_str = get_article_ids(elem, json_str)					
					json_str['citations_pmid'] = get_article_citations(elem)


					annotation_dict = get_abstract_conceptids_2(json_str, article_text, cursor)
					abstract_sentences = None
					if annotation_dict['abstract'] is not None:
						json_str['abstract_conceptids'] = annotation_dict['abstract']['cid_dict']
						json_str['abstract_dids'] = annotation_dict['abstract']['did_dict']
						abstract_sentences = annotation_dict['abstract']['sentences']
					else:
						json_str['abstract_conceptids'] = None
						json_str['abstract_dids'] = None
					
					if annotation_dict['title'] is not None:
						title_sentences = annotation_dict['title']['sentences']
						title_cids = annotation_dict['title']['cids']
						title_dids = annotation_dict['title']['dids']
					else:
						title_sentences = None
						title_cids = None
						title_dids = None

					if title_sentences is not None:
						title_sentences['pmid'] = json_str['pmid']

					if title_cids is not None:
						json_str['title_conceptids'] = title_cids
						json_str['title_dids'] = title_dids
					else:
						json_str['title_conceptids'] = None
						json_str['title_dids'] = None

					# json_str['abstract_conceptids'], json_str['abstract_dids'], abstract_sentences = get_abstract_conceptids_2(json_str, article_text, cursor)
					
					s = pd.DataFrame(columns=['id', 'conceptid', 'concept_arr', 'section', 'line_num', 'sentence', 'sentence_tuples', 'section_index', 'pmid'])
					if title_sentences is not None:
						s = s.append(title_sentences, sort=False)
					if abstract_sentences is not None:
						abstract_sentences['pmid'] = json_str['pmid']
						s = s.append(abstract_sentences, sort=False)
					if len(s) > 0:
						s = s[['id', 'pmid', 'conceptid', 'concept_arr', 'section', 'section_index', 'line_num', 'sentence', 'sentence_tuples']]
						u.write_sentences(s, cursor)


					json_str['index_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
				

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
							raise ValueError('incompatible json obj')
						
						
					elif query_result['hits']['total'] == 1:
						article_id = query_result['hits']['hits'][0]['_id']
						es.index(index=INDEX_NAME, id=article_id, doc_type='abstract', body=json_obj)
					# cursor.close()
					# conn.close()



def load_pubmed_local_2(start_file):
	es = u.get_es_client()
	number_of_processes = 8

	cursors = []
	for i in range(number_of_processes):
		conn, cursor = pg.return_postgres_cursor()
		cursors.append((conn,cursor))

	task_queue = mp.Queue()

	pool = []
	for i in range(number_of_processes):
		p = mp.Process(target=doc_worker, args=(task_queue, cursors[i][0], cursors[i][1]))
		pool.append(p)
		p.start()

	conn, cursor = pg.return_postgres_cursor()
	issn_query = "select issn from pubmed.journals"
	issn_list = pg.return_df_from_query(cursor, issn_query, None, ['issn'])['issn'].tolist()
	cursor.close()
	conn.close()

	index_exists = es.indices.exists(index=INDEX_NAME)
	if not index_exists:
		settings = {"mappings" : {"abstract" : {"properties" : {
			"journal_issn" : {"type" : "keyword"}
			,"journal_issn_type" : {
				"properties" : {"IssnType" : {"type" : "keyword"}}
				}
			,"journal_title" : {"type" : "text"}
			,"journal_iso_abbrev" : {"type" : "keyword"}
			,"journal_volume" : {"type" : "text"}
			,"journal_issue" : {"type" : "text"}
			,"journal_pub_year" : {"type" : "integer"}
			,"journal_pub_month" : {"type" : "keyword"}
			,"journal_pub_day" : {"type" : "keyword"}
			,"journal_issue" : {"type" : "keyword"}
			,"article_title" : {"type" : "text"}
			,"article_abstract" : {"properties" : {}}
			,"pmid" : {"type" : "integer"}
			,"article_ids" : {"properties" : {"pmid" : {"type" : "keyword"}, 
				"doi" : {"type" : "keyword"},
				"pii" : {"type" : "keyword"}}}
			,"citations_pmid" : {"type" : "keyword"}
			,"title_conceptids" : {"type" : "keyword"}
			,"title_dids" : {"type" : "keyword"}
			,"abstract_conceptids" : {"properties" : {"methods_cid" : {"type" : "keyword"}, 
				"background_cid" : {"type" : "keyword"},
				"conclusions_cid" : {"type" : "keyword"},
				"objective_cid" : {"type" : "keyword"},
				"results_cid" : {"type" : "keyword"},
				"unlabelled_cid" : {"type" : "keyword"}}}
			,"abstract_dids" : {"properties" : {"methods_did" : {"type" : "keyword"}, 
				"background_did" : {"type" : "keyword"},
				"conclusions_did" : {"type" : "keyword"},
				"objective_did" : {"type" : "keyword"},
				"results_did" : {"type" : "keyword"}}}
			,"article_type_id" : {"type" : "keyword"}
			,"article_type" : {"type" : "keyword"}
			,"index_date" : {"type" : "date", "format": "yyyy-MM-dd HH:mm:ss"}
			,"filename" : {"type" : "keyword"}
		}}}}
		es.indices.create(index=INDEX_NAME, body=settings)
		

	folder_arr = ['resources/updatefiles']

	for folder_path in folder_arr:
		file_counter = 0

		file_lst = os.listdir(folder_path)
		file_lst.sort()

		for filename in file_lst:

			abstract_counter = 0
			file_path = folder_path + '/' + filename
			file_num = int(re.findall('pubmed20n(.*).xml', filename)[0])

			if file_num >= start_file:
				print(filename)
			
				file_timer = u.Timer('file')

				# tree = ET.parse(file_path)		
				# root = tree.getroot()
				file_abstract_counter = 0
				for event, elem in ET.iterparse(file_path, tag="PubmedArticle"):
					json_str = {}
					params = (ET.tostring(elem), filename, issn_list)
					task_queue.put((index_doc_from_elem, params))
					file_abstract_counter += 1
					elem.clear()


				file_timer.stop()
				if file_num >= start_file+10:
					break
				
		if file_num == start_file+10:
			break
				
				

	for i in range(number_of_processes):
		task_queue.put('STOP')

	for p in pool:
		p.join()
	

def load_pubmed_local_test(start_file):
	es = u.get_es_client()
	number_of_processes = 8
	
	

	cursors = []
	for i in range(number_of_processes):
		conn, cursor = pg.return_postgres_cursor()
		cursors.append((conn,cursor))

	task_queue = mp.Queue()

	pool = []
	for i in range(number_of_processes):
		p = mp.Process(target=doc_worker, args=(task_queue,cursors[i][0], cursors[i][1]))
		pool.append(p)
		p.start()

	# task_queue = mp.Queue()
	# pool = []
	# for i in range(number_of_processes):
	# 	p = mp.Process(target=doc_worker, args=(task_queue,))
	# 	pool.append(p)
	# 	p.start()

	index_exists = es.indices.exists(index=INDEX_NAME)
	if not index_exists:
		settings = {"mappings" : {"abstract" : {"properties" : {
			"journal_issn" : {"type" : "keyword"}
			,"journal_issn_type" : {
				"properties" : {"IssnType" : {"type" : "keyword"}}
				}
			,"journal_title" : {"type" : "text"}
			,"journal_iso_abbrev" : {"type" : "keyword"}
			,"journal_volume" : {"type" : "text"}
			,"journal_issue" : {"type" : "text"}
			,"journal_pub_year" : {"type" : "integer"}
			,"journal_pub_month" : {"type" : "keyword"}
			,"journal_pub_day" : {"type" : "keyword"}
			,"journal_issue" : {"type" : "keyword"}
			,"article_title" : {"type" : "text"}
			,"article_abstract" : {"properties" : {}}
			,"pmid" : {"type" : "integer"}
			,"article_ids" : {"properties" : {"pmid" : {"type" : "keyword"}, 
				"doi" : {"type" : "keyword"},
				"pii" : {"type" : "keyword"}}}
			,"citations_pmid" : {"type" : "keyword"}
			,"title_conceptids" : {"type" : "keyword"}
			,"title_dids" : {"type" : "keyword"}
			,"abstract_conceptids" : {"properties" : {"methods_cid" : {"type" : "keyword"}, 
				"background_cid" : {"type" : "keyword"},
				"conclusions_cid" : {"type" : "keyword"},
				"objective_cid" : {"type" : "keyword"},
				"results_cid" : {"type" : "keyword"},
				"unlabelled_cid" : {"type" : "keyword"}}}
			,"abstract_dids" : {"properties" : {"methods_did" : {"type" : "keyword"}, 
				"background_did" : {"type" : "keyword"},
				"conclusions_did" : {"type" : "keyword"},
				"objective_did" : {"type" : "keyword"},
				"results_did" : {"type" : "keyword"}}}
			,"article_type_id" : {"type" : "keyword"}
			,"article_type" : {"type" : "keyword"}
			,"index_date" : {"type" : "date", "format": "yyyy-MM-dd HH:mm:ss"}
			,"filename" : {"type" : "keyword"}
		}}}}
		es.indices.create(index=INDEX_NAME, body=settings)
		

	folder_path = 'resources/baseline'


	file_counter = 0

	file_lst = os.listdir(folder_path)
	file_lst.sort()
	for filename in file_lst:
		abstract_counter = 0
		file_path = folder_path + '/' + filename
		file_num = int(re.findall('pubmed18n(.*).xml', filename)[0])
		if file_num == start_file:
			print(filename)
		
			file_timer = u.Timer('file')
			# tree = ET.parse(file_path)		
			# root = tree.getroot()
			file_abstract_counter = 0
			for event, elem in ET.iterparse(file_path, tag="PubmedArticle"):
				json_str = {}
				params = (ET.tostring(elem), filename)
				task_queue.put((index_doc_from_elem, params))
				file_abstract_counter += 1
				elem.clear()
			file_timer.stop()
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

def get_abstract_conceptids_3(abstract_dict, article_text, cursor):
	cid_dict = {}
	did_dict = {}
	result_dict = {}
	cleaned_text = ann.clean_text(article_text)
	all_words = ann.get_all_words_list(cleaned_text)
	cache = ann.get_cache(all_words, True, cursor)

	prelim_ann = pd.DataFrame()
	res, new_sentences = get_snomed_annotation(abstract_dict['article_title'], 'title', cache, cursor)
	prelim_ann = prelim_ann.append(res, sort=False)

	if abstract_dict['article_abstract'] is not None:
		for index,k1 in enumerate(abstract_dict['article_abstract']):
			res,new_sentences = get_snomed_annotation(abstract_dict['article_abstract'][k1], str(k1), cache, cursor)

			if new_sentences is not None:
				new_sentences['section_index'] = index
				sentences = sentences.append(new_sentences, sort=False)
			prelim_ann = prelim_ann.append(res, sort=False)

		prelim_ann = ann.acronym_check(prelim_ann)

		for index,k1 in enumerate(abstract_dict['article_abstract']):
			k1_cid = str(k1) + "_cid"
			k1_did = str(k1) + "_did"

			annotated = prelim_ann[prelim_ann['section'] == str(k1)]
			if annotated is not None:
				cid_dict[k1_cid] = annotated['conceptid'].tolist()
				did_dict[k1_did] = annotated['description_id'].tolist()
			else:
				cid_dict[k1_cid] = None
				did_dict[k1_did] = None
		result_dict['abstract'] = {'cid_dict' : cid_dict, 'did_dict' : did_dict, 'sentences' : sentences}
		return result_dict
	else:
		result_dict['abstract'] = None
		return result_dict


def get_abstract_conceptids_2(abstract_dict, article_text, cursor):
	cid_dict = {}
	did_dict = {}
	result_dict = {}
	
	cleaned_text = ann.clean_text(article_text)
	all_words = ann.get_all_words_list(cleaned_text)
	
	# True = case_sensitive
	cache = ann.get_cache(all_words, True, cursor)

	res, new_sentences = get_snomed_annotation(abstract_dict['article_title'], 'title', cache, cursor)
	if res is not None:
		result_dict['title'] = {'cids' : res['conceptid'].tolist(), 'dids' : res['description_id'].tolist(), 'sentences' : new_sentences}
	else:
		result_dict['title'] = None

	sentences = pd.DataFrame(columns=['id', 'conceptid', 'concept_arr', 'section', 'line_num', 'sentence', 'sentence_tuples'])
	if abstract_dict['article_abstract'] is not None:
		for index,k1 in enumerate(abstract_dict['article_abstract']):
			res,new_sentences = get_snomed_annotation(abstract_dict['article_abstract'][k1], str(k1), cache, cursor)
			
			if new_sentences is not None:
				new_sentences['section_index'] = index	
				sentences = sentences.append(new_sentences, sort=False)
				

			k1_cid = str(k1) + "_cid"
			k1_did = str(k1) + "_did"
			if res is not None:
				cid_dict[k1_cid] = res['conceptid'].tolist()
				did_dict[k1_did] = res['description_id'].tolist()
			else:
				cid_dict[k1_cid] = None
				did_dict[k1_did] = None
		result_dict['abstract'] = {'cid_dict' : cid_dict, 'did_dict' : did_dict, 'sentences' : sentences}
		return result_dict
	else:
		result_dict['abstract'] = None
		return result_dict

def get_deleted_pmid(elem):
	delete_pmid_arr = []
	for item in elem:
		delete_pmid_arr.append(str(item.text))

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
							citation_pmid_list.append(str(item.text))
		return citation_pmid_list
	else:
		return None

def get_article_ids(elem, json_str):
	id_list_elem = elem.find('*/ArticleIdList')
	article_id_dict = {}
	for elem_id in id_list_elem:
		if elem_id.attrib['IdType'] == 'pubmed':
			article_id_dict['pmid'] = str(elem_id.text)
		else:
			article_id_dict[elem_id.attrib['IdType']] = str(elem_id.text)
	json_str['article_ids'] = article_id_dict

	return json_str

def is_issn(elem, issn):
	try:
		j_list = elem.findall('./MedlineCitation/Article/Journal')
		journal_elem = j_list[0]
		issn_elem = journal_elem.findall('./ISSN')[0]
		issn_text = str(issn_elem.text)

		if issn_text == issn:
			return True
		else:
			return False
	except:
		return False

def return_issn(elem):
	try:
		j_list = elem.findall('./MedlineCitation/Article/Journal')
		journal_elem = j_list[0]
		issn_elem = journal_elem.findall('./ISSN')[0]
		issn_text = str(issn_elem.text)
		return issn_text
		
	except:
		return '0'

def get_pmid(elem, json_str):
	pmid = elem.findall('*/PMID')
	try:
		json_str['pmid'] = str(pmid[0].text)
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
		issn_elem = str(journal_elem.findall('./ISSN')[0])
		json_str['journal_issn'] = str(issn_elem.text)
		json_str['journal_issn_type'] = str(issn_elem.attrib)
	except:
		json_str['journal_issn'] = None
		json_str['journal_issn_type'] = None

	try:
		title_elem = journal_elem.findall('./Title')[0]
		json_str['journal_title'] = str(title_elem.text)
	except:
		json_str['journal_title'] = None

	try:
		iso_elem = journal_elem.find('./ISOAbbreviation')
		json_str['journal_iso_abbrev'] = str(iso_elem.text)
	except:
		json_str['journal_iso_abbrev'] = None

	try:
		journal_volume_elem = journal_elem.find('./JournalIssue/Volume')
		json_str['journal_volume'] = str(journal_volume_elem.text)
	except:
		json_str['journal_volume'] = None

	try:
		journal_issue_elem = journal_elem.find('./JournalIssue/Issue')
		json_str['journal_issue'] = str(journal_issue_elem.text)
	except:
		json_str['journal_issue'] = None

	try:
		year_elem = journal_elem.findall('./JournalIssue/PubDate/Year')[0]
		json_str['journal_pub_year'] = str(year_elem.text)
	except:
		json_str['journal_pub_year'] = None

	try:
		month_elem = journal_elem.findall('./JournalIssue/PubDate/Month')[0]
		json_str['journal_pub_month'] = str(month_elem.text)
	except:
		json_str['journal_pub_month'] = None

	try:
		day_elem = journal_elem.findall('./JournalIssue/PubDate/Day')[0]
		day_str['journal_pub_month'] = str(day_elem.text)
	except:
		json_str['journal_pub_day'] = None

	return json_str


def get_article_info_2(elem, json_str):
	article_text = ""
	try:
		article_elem = elem.find('./MedlineCitation/Article')
	except:
		json_str['article_title'] = None
		json_str['article_abstract'] = None
		json_str['article_type'] = None
		json_str['article_type_id'] = None

	try:
		title_elem = article_elem.find('./ArticleTitle')
		json_str['article_title'] = str(title_elem.text)
		article_text += str(title_elem.text)
	except:
		json_str['article_title'] = None

	try:
		abstract_elem = article_elem.find('./Abstract')
		abstract_dict = {}

		for abstract_sub_elem in abstract_elem:

			if not abstract_sub_elem.attrib:

				if 'unlabelled' not in abstract_dict.keys():
					abstract_dict['unlabelled'] = str(abstract_sub_elem.text)
				else:
					abstract_dict['unlabelled'] = abstract_dict['unlabelled'] + "\r" + str(abstract_sub_elem.text)
				article_text += ' ' + str(abstract_sub_elem.text)
			else:			
				try:
					if abstract_sub_elem.attrib['NlmCategory'].lower() in abstract_dict.keys():
						abstract_dict[abstract_sub_elem.attrib['NlmCategory'].lower()] = \
							abstract_dict[abstract_sub_elem.attrib['NlmCategory'].lower()] + "\r" + str(abstract_sub_elem.text)
					else:
						abstract_dict[abstract_sub_elem.attrib['NlmCategory'].lower()] = str(abstract_sub_elem.text)
					article_text += ' ' + str(abstract_sub_elem.text)
				except:
					
					try: 
						if nlm_cat_dict[abstract_sub_elem.attrib['Label'].lower()] in abstract_dict.keys():
							abstract_dict[nlm_cat_dict[abstract_sub_elem.attrib['Label']].lower()] = \
								abstract_dict[nlm_cat_dict[abstract_sub_elem.attrib['Label']].lower()] + "\r" + str(abstract_sub_elem.text)
						else:
							abstract_dict[nlm_cat_dict[abstract_sub_elem.attrib['Label'].lower()]] = str(abstract_sub_elem.text)
						article_text += ' ' + str(abstract_sub_elem.text)
					except:
						if 'unlabelled' not in abstract_dict.keys():
							abstract_dict['unlabelled'] = str(abstract_sub_elem.text)
						else:
							abstract_dict['unlabelled'] = abstract_dict['unlabelled'] + "\r" + str(abstract_sub_elem.text)
						article_text += ' ' + str(abstract_sub_elem.text)

		json_str['article_abstract'] = abstract_dict

	except:
		json_str['article_abstract'] = None

	try:
		article_type_elem = article_elem.findall('./PublicationTypeList/PublicationType')
		json_str['article_type'] = []
		json_str['article_type_id'] = []

		for node in article_type_elem:
			json_str['article_type'].append(str(node.text))
			json_str['article_type_id'].append(str(node.attrib['UI']))
	except:
		json_str['article_type'] = None
		json_str['article_type_id'] = None

	return json_str, article_text

def get_snomed_annotation(text, section, cache, cursor):
	
	if text is None:
		return None, None
	else:
		annotation, sentences = ann.annotate_text_not_parallel(text, section, cache, cursor, True, True, True)
		if not annotation.empty:
			return annotation, sentences
		else:
			return None, None

def index_sentences(index_num):
	conn, cursor = pg.return_postgres_cursor()

	index_query = """
		set schema 'annotation';
		create index sentences5_id_ind on sentences5(id);
		create index sentences5_pmid_ind on sentences5(pmid);
		create index sentences5_conceptid_ind on sentences5(conceptid);
		create index sentences5_section_ind on sentences5(section);
	"""

	cursor.execute(index_query, None)
	cursor.connection.commit()
	cursor.close()		

if __name__ == "__main__":

	# start_file = 700
	# c = u.Timer("full timer")
	# load_pubmed_local_test(364)
	# c.stop()


	start_file = 1016
	while (start_file < 1132):
		print(start_file)
		load_pubmed_local_2(start_file)
		start_file += 10
	# index_sentences(5)
	

nlm_cat_dict = {"a case report" :"methods"
,"abbreviations" :"background"
,"access to data" :"background"
,"accme accreditation" :"background"
,"achievements" :"results"
,"acknowledgments" :"background"
,"acquisition of evidence" :"methods"
,"action" :"methods"
,"actions" :"methods"
,"actions taken" :"conclusions"
,"activities" :"methods"
,"admission findings" :"methods"
,"advances in knowledge" :"conclusions"
,"advances in knowledge and implications for patient care" :"conclusions"
,"advantages" :"conclusions"
,"adverse effects" :"results"
,"aetiology" :"background"
,"aim" :"objective"
,"aim & design" :"objective"
,"aim & method" :"objective"
,"aim & objectives" :"objective"
,"aim & purpose" :"objective"
,"aim and background" :"objective"
,"aim and design" :"objective"
,"aim and goals" :"objective"
,"aim and materials & methods" :"objective"
,"aim and method" :"objective"
,"aim and methodology" :"objective"
,"aim and methods" :"objective"
,"aim and objective" :"objective"
,"aim and objectives" :"objective"
,"aim and purpose" :"objective"
,"aim and scope" :"objective"
,"aim and setting" :"objective"
,"aim of investigation" :"objective"
,"aim of paper" :"objective"
,"aim of study" :"objective"
,"aim of the paper" :"objective"
,"aim of the review" :"objective"
,"aim of the study" :"objective"
,"aim of the study and methods" :"objective"
,"aim of the work" :"objective"
,"aim of this study" :"objective"
,"aim of work" :"objective"
,"aim(s)" :"objective"
,"aim(s) of the study" :"objective"
,"aim, patients and methods" :"objective"
,"aim/background" :"objective"
,"aim/hypothesis" :"objective"
,"aim/method" :"objective"
,"aim/methods" :"objective"
,"aim/objective" :"objective"
,"aim/objectives" :"objective"
,"aim/purpose" :"objective"
,"aims" :"objective"
,"aims & background" :"objective"
,"aims & method" :"objective"
,"aims & methods" :"objective"
,"aims & objectives" :"objective"
,"aims & results" :"objective"
,"aims and background" :"objective"
,"aims and design" :"objective"
,"aims and development" :"objective"
,"aims and hypothesis" :"objective"
,"aims and method" :"objective"
,"aims and methodology" :"objective"
,"aims and methods" :"objective"
,"aims and objective" :"objective"
,"aims and objectives" :"objective"
,"aims and purpose" :"objective"
,"aims and scope" :"objective"
,"aims and/or hypothesis" :"objective"
,"aims of study" :"objective"
,"aims of the paper" :"objective"
,"aims of the review" :"objective"
,"aims of the study" :"objective"
,"aims of this study" :"objective"
,"aims/background" :"objective"
,"aims/hypotheses" :"objective"
,"aims/hypothesis" :"objective"
,"aims/introduction" :"objective"
,"aims/methods" :"objective"
,"aims/methods & procedures" :"objective"
,"aims/objective" :"objective"
,"aims/objectives" :"objective"
,"aims/purpose" :"objective"
,"ams subject classification" :"methods"
,"analyses" :"methods"
,"analysis" :"methods"
,"analysis and results" :"results"
,"analysis of results" :"results"
,"analytic validity" :"results"
,"analytical approach" :"methods"
,"analytical techniques" :"methods"
,"anamnesis" :"background"
,"anamnesis and clinical findings" :"background"
,"anatomy" :"methods"
,"anesthesia" :"methods"
,"animal" :"methods"
,"animal or sample population" :"methods"
,"animal population" :"methods"
,"animal studied" :"methods"
,"animal studies" :"methods"
,"animal(s)" :"methods"
,"animals" :"methods"
,"animals and interventions" :"methods"
,"animals and methods" :"methods"
,"animals or sample population" :"methods"
,"animals studied" :"methods"
,"animals, materials and methods" :"methods"
,"animals, methods" :"methods"
,"answer" :"conclusions"
,"antecedents" :"background"
,"application" :"conclusions"
,"application to clinical practice" :"conclusions"
,"application/conclusions" :"conclusions"
,"applications" :"conclusions"
,"applications/conclusion" :"conclusions"
,"applications/conclusions" :"conclusions"
,"approach" :"methods"
,"approach and methods" :"methods"
,"approach and results" :"results"
,"approach to the problem" :"background"
,"area covered" :"methods"
,"area covered in this review" :"methods"
,"areas covered" :"methods"
,"areas covered in the review" :"methods"
,"areas covered in this review" :"methods"
,"areas of agreement" :"results"
,"areas of agreement and controversy" :"results"
,"areas of controversy" :"results"
,"areas timely for developing research" :"conclusions"
,"areas to develop research" :"conclusions"
,"argument" :"objective"
,"arguments" :"objective"
,"article chosen" :"methods"
,"article selection" :"methods"
,"article title and bibliographic information" :"background"
,"assessment" :"results"
,"assessment of problem" :"methods"
,"assessment of risk factors" :"methods"
,"assessment tools" :"methods"
,"assessments" :"methods"
,"audience" :"background"
,"author's conclusions" :"conclusions"
,"authors' conclusion" :"conclusions"
,"authors' conclusions" :"conclusions"
,"availability" :"background"
,"availability and implementation" :"methods"
,"availability and supplementary information" :"background"
,"background" :"background"
,"background & aim" :"objective"
,"background & aims" :"objective"
,"background & goals" :"objective"
,"background & materials and methods" :"methods"
,"background & method" :"methods"
,"background & methods" :"methods"
,"background & objective" :"objective"
,"background & objectives" :"objective"
,"background & problems" :"objective"
,"background & purpose" :"objective"
,"background & rationale" :"objective"
,"background aims" :"objective"
,"background and aim" :"objective"
,"background and aim of study" :"objective"
,"background and aim of the study" :"objective"
,"background and aim of the work" :"objective"
,"background and aim of work" :"objective"
,"background and aims" :"objective"
,"background and aims of the study" :"objective"
,"background and context" :"background"
,"background and design" :"methods"
,"background and goal" :"objective"
,"background and goals" :"objective"
,"background and hypothesis" :"objective"
,"background and importance" :"background"
,"background and introduction" :"background"
,"background and method" :"methods"
,"background and methodology" :"methods"
,"background and methods" :"methods"
,"background and motivation" :"background"
,"background and object" :"objective"
,"background and objective" :"objective"
,"background and objectives" :"objective"
,"background and overview" :"background"
,"background and procedure" :"methods"
,"background and purpose" :"objective"
,"background and purpose of study" :"objective"
,"background and purpose of the study" :"objective"
,"background and purposes" :"objective"
,"background and rationale" :"objective"
,"background and research objective" :"objective"
,"background and research objectives" :"objective"
,"background and results" :"results"
,"background and scope" :"objective"
,"background and setting" :"methods"
,"background and significance" :"background"
,"background and study aim" :"objective"
,"background and study aims" :"objective"
,"background and study objective" :"objective"
,"background and study objectives" :"objective"
,"background and study purpose" :"objective"
,"background and the purpose of the study" :"objective"
,"background content" :"background"
,"background context" :"background"
,"background data" :"background"
,"background data and objective" :"objective"
,"background information" :"background"
,"background objectives" :"objective"
,"background of the study" :"background"
,"background purpose" :"objective"
,"background to the debate" :"background"
,"background, aim and scope" :"objective"
,"background, aim, and scope" :"objective"
,"background, aims" :"objective"
,"background, aims & methods" :"objective"
,"background, aims and scope" :"objective"
,"background, aims, and scope" :"objective"
,"background, material and methods" :"methods"
,"background, materials and methods" :"methods"
,"background/aim" :"objective"
,"background/aim of study" :"objective"
,"background/aims" :"objective"
,"background/aims and methods" :"objective"
,"background/aims/methods" :"objective"
,"background/goals" :"objective"
,"background/hypothesis" :"objective"
,"background/introduction" :"background"
,"background/method" :"methods"
,"background/methods" :"methods"
,"background/objective" :"objective"
,"background/objectives" :"objective"
,"background/purpose" :"objective"
,"background/purpose(s)" :"objective"
,"background/purposes" :"objective"
,"background/rationale" :"background"
,"background/significance" :"background"
,"background/study context" :"objective"
,"backgrounds" :"background"
,"backgrounds & aims" :"objective"
,"backgrounds and aim" :"objective"
,"backgrounds and aims" :"objective"
,"backgrounds and objective" :"objective"
,"backgrounds and objectives" :"objective"
,"backgrounds and purpose" :"objective"
,"backgrounds/aims" :"objective"
,"baseline data" :"methods"
,"baseline results" :"results"
,"basic design" :"methods"
,"basic methods" :"methods"
,"basic problem" :"objective"
,"basic problem and aim of study" :"objective"
,"basic problem and objective" :"objective"
,"basic problem and objective of study" :"objective"
,"basic problem and objective of the study" :"objective"
,"basic problems and objective" :"objective"
,"basic procedure" :"methods"
,"basic procedures" :"methods"
,"basic remarks" :"background"
,"basic research design" :"methods"
,"basic research design and participants" :"methods"
,"basics" :"background"
,"basis" :"background"
,"benefits" :"results"
,"benefits, harm and costs" :"results"
,"benefits, harms and costs" :"results"
,"benefits, harms, and costs" :"results"
,"benefits, harms, costs" :"results"
,"best practices" :"conclusions"
,"bias, confounding and other reasons for caution" :"conclusions"
,"blinding" :"methods"
,"bottom line" :"conclusions"
,"brief description" :"methods"
,"cadavers" :"methods"
,"calculation" :"methods"
,"calculations" :"methods"
,"capsule summary" :"conclusions"
,"case" :"methods"
,"case 1" :"methods"
,"case 2" :"methods"
,"case and methods" :"methods"
,"case characteristics" :"methods"
,"case definition" :"methods"
,"case description" :"methods"
,"case description and methods" :"methods"
,"case descriptions" :"methods"
,"case details" :"methods"
,"case diagnosis/treatment" :"methods"
,"case discussion" :"methods"
,"case histories" :"methods"
,"case history" :"methods"
,"case history and clinical findings" :"methods"
,"case history and diagnosis" :"methods"
,"case history and findings" :"methods"
,"case illustration" :"methods"
,"case management" :"results"
,"case material" :"methods"
,"case or series summary" :"methods"
,"case outline" :"methods"
,"case outlines" :"methods"
,"case presentation" :"methods"
,"case presentation & methods" :"methods"
,"case presentation and intervention" :"methods"
,"case presentations" :"methods"
,"case record" :"methods"
,"case report" :"methods"
,"case report and discussion" :"methods"
,"case report and method" :"methods"
,"case report and methods" :"methods"
,"case report and results" :"results"
,"case report/methods" :"methods"
,"case reports" :"methods"
,"case results" :"methods"
,"case review" :"methods"
,"case series" :"methods"
,"case series summary" :"methods"
,"case studies" :"methods"
,"case study" :"methods"
,"case summaries" :"methods"
,"case summary" :"methods"
,"case(s)" :"methods"
,"case-diagnosis/treatment" :"methods"
,"case-report" :"methods"
,"case-reports" :"methods"
,"cases" :"methods"
,"cases & methods" :"methods"
,"cases and method" :"methods"
,"cases and methods" :"methods"
,"cases description" :"methods"
,"cases presentation" :"methods"
,"cases report" :"methods"
,"cases reports" :"methods"
,"casuistic & methods" :"methods"
,"casuistic and method" :"methods"
,"casuistic and methods" :"methods"
,"centers" :"methods"
,"challenges" :"conclusions"
,"challenges and lessons learned" :"conclusions"
,"chemotherapy" :"methods"
,"chief outcome measures" :"methods"
,"children and methods" :"methods"
,"choice of solution" :"methods"
,"citation" :"background"
,"classification" :"methods"
,"classification of evidence" :"methods"
,"clinic case" :"methods"
,"clinical" :"methods"
,"clinical advantages" :"conclusions"
,"clinical and pathological findings" :"results"
,"clinical application" :"methods"
,"clinical applications" :"methods"
,"clinical aspects" :"methods"
,"clinical case" :"methods"
,"clinical cases" :"methods"
,"clinical challenges" :"objective"
,"clinical characteristics" :"methods"
,"clinical context" :"methods"
,"clinical course" :"methods"
,"clinical data" :"methods"
,"clinical description" :"methods"
,"clinical feature" :"methods"
,"clinical features" :"methods"
,"clinical findings" :"results"
,"clinical findings and diagnosis" :"results"
,"clinical history" :"methods"
,"clinical implication" :"conclusions"
,"clinical implications" :"conclusions"
,"clinical importance" :"conclusions"
,"clinical investigations" :"methods"
,"clinical issue" :"objective"
,"clinical manifestations" :"results"
,"clinical material" :"methods"
,"clinical material and methods" :"methods"
,"clinical materials and methods" :"methods"
,"clinical need" :"objective"
,"clinical observation" :"methods"
,"clinical observations" :"methods"
,"clinical outcome" :"results"
,"clinical picture" :"methods"
,"clinical potential" :"conclusions"
,"clinical presentation" :"methods"
,"clinical presentation and intervention" :"methods"
,"clinical presentations" :"methods"
,"clinical problem" :"objective"
,"clinical procedure" :"methods"
,"clinical question" :"objective"
,"clinical question/level of evidence" :"methods"
,"clinical recommendations" :"conclusions"
,"clinical rehabilitation impact" :"conclusions"
,"clinical relevance" :"conclusions"
,"clinical report" :"methods"
,"clinical results" :"results"
,"clinical scenario" :"methods"
,"clinical setting" :"methods"
,"clinical settings" :"methods"
,"clinical significance" :"conclusions"
,"clinical signs" :"methods"
,"clinical studies" :"methods"
,"clinical study" :"methods"
,"clinical symptoms" :"methods"
,"clinical trial" :"background"
,"clinical trial information" :"background"
,"clinical trial number" :"background"
,"clinical trial registration" :"background"
,"clinical trial registration information" :"background"
,"clinical trial registration number" :"background"
,"clinical trial registration url" :"background"
,"clinical trial registration- url" :"background"
,"clinical trial registry" :"background"
,"clinical trial registry number" :"background"
,"clinical trials" :"results"
,"clinical trials identifier" :"background"
,"clinical trials registration" :"background"
,"clinical trials registration number" :"background"
,"clinical trials registry" :"background"
,"clinical utility" :"results"
,"clinical validity" :"results"
,"clinical value" :"conclusions"
,"clinical/methodical issue" :"objective"
,"cohort" :"methods"
,"comment" :"conclusions"
,"commentaries" :"conclusions"
,"commentary" :"conclusions"
,"comments" :"conclusions"
,"comments and conclusion" :"conclusions"
,"comments and conclusions" :"conclusions"
,"community context" :"objective"
,"comparators" :"methods"
,"comparison with existing method" :"conclusions"
,"comparison with existing method(s)" :"conclusions"
,"comparison with existing methods" :"methods"
,"complications" :"results"
,"concept" :"methods"
,"conceptual framework" :"methods"
,"concluding remarks" :"conclusions"
,"concluding statement" :"conclusions"
,"conclusion" :"conclusions"
,"conclusion & clinical relevance" :"conclusions"
,"conclusion & discussion" :"conclusions"
,"conclusion & implications" :"conclusions"
,"conclusion & inferences" :"conclusions"
,"conclusion and clinical implications" :"conclusions"
,"conclusion and clinical importance" :"conclusions"
,"conclusion and clinical rehabilitation impact" :"conclusions"
,"conclusion and clinical relevance" :"conclusions"
,"conclusion and clinical significance" :"conclusions"
,"conclusion and discussion" :"conclusions"
,"conclusion and general significance" :"conclusions"
,"conclusion and impact" :"conclusions"
,"conclusion and implication" :"conclusions"
,"conclusion and implications" :"conclusions"
,"conclusion and implications for practice" :"conclusions"
,"conclusion and message" :"conclusions"
,"conclusion and outlook" :"conclusions"
,"conclusion and perspective" :"conclusions"
,"conclusion and perspectives" :"conclusions"
,"conclusion and potential relevance" :"conclusions"
,"conclusion and practice implications" :"conclusions"
,"conclusion and recommendation" :"conclusions"
,"conclusion and recommendations" :"conclusions"
,"conclusion and relevance" :"conclusions"
,"conclusion and relevance to clinical practice" :"conclusions"
,"conclusion and scientific significance" :"conclusions"
,"conclusion and significance" :"conclusions"
,"conclusion(s)" :"conclusions"
,"conclusion-discussion" :"conclusions"
,"conclusion/clinical relevance" :"conclusions"
,"conclusion/discussion" :"conclusions"
,"conclusion/hypothesis" :"conclusions"
,"conclusion/implications" :"conclusions"
,"conclusion/implications for practice" :"conclusions"
,"conclusion/interpretation" :"conclusions"
,"conclusion/recommendation" :"conclusions"
,"conclusion/recommendations" :"conclusions"
,"conclusion/relevance" :"conclusions"
,"conclusion/significance" :"conclusions"
,"conclusions" :"conclusions"
,"conclusions & clinical importance" :"conclusions"
,"conclusions & clinical relevance" :"conclusions"
,"conclusions & implication" :"conclusions"
,"conclusions & implications" :"conclusions"
,"conclusions & implications for nursing" :"conclusions"
,"conclusions & inferences" :"conclusions"
,"conclusions & interpretation" :"conclusions"
,"conclusions & recommendations" :"conclusions"
,"conclusions & significance" :"conclusions"
,"conclusions / implications for practice" :"conclusions"
,"conclusions and clinical implications" :"conclusions"
,"conclusions and clinical importance" :"conclusions"
,"conclusions and clinical relevance" :"conclusions"
,"conclusions and clinical significance" :"conclusions"
,"conclusions and discussion" :"conclusions"
,"conclusions and general significance" :"conclusions"
,"conclusions and impact" :"conclusions"
,"conclusions and implication" :"conclusions"
,"conclusions and implications" :"conclusions"
,"conclusions and implications for cancer survivors" :"conclusions"
,"conclusions and implications for nursing management" :"conclusions"
,"conclusions and implications for practice" :"conclusions"
,"conclusions and implications of key findings" :"conclusions"
,"conclusions and inferences" :"conclusions"
,"conclusions and interpretation" :"conclusions"
,"conclusions and limitations" :"conclusions"
,"conclusions and outlook" :"conclusions"
,"conclusions and perspectives" :"conclusions"
,"conclusions and potential relevance" :"conclusions"
,"conclusions and practical implications" :"conclusions"
,"conclusions and practice implications" :"conclusions"
,"conclusions and recommendations" :"conclusions"
,"conclusions and relevance" :"conclusions"
,"conclusions and relevance to clinical practice" :"conclusions"
,"conclusions and scientific significance" :"conclusions"
,"conclusions and significance" :"conclusions"
,"conclusions significance" :"conclusions"
,"conclusions(s)" :"conclusions"
,"conclusions, significance and impact of the study" :"conclusions"
,"conclusions/ significance" :"conclusions"
,"conclusions/applications" :"conclusions"
,"conclusions/clinical relevance" :"conclusions"
,"conclusions/discussion" :"conclusions"
,"conclusions/findings" :"conclusions"
,"conclusions/implications" :"conclusions"
,"conclusions/implications for practice" :"conclusions"
,"conclusions/importance" :"conclusions"
,"conclusions/interpretation" :"conclusions"
,"conclusions/interpretations" :"conclusions"
,"conclusions/lessons learned" :"conclusions"
,"conclusions/level of evidence" :"conclusions"
,"conclusions/practice implications" :"conclusions"
,"conclusions/recommendations" :"conclusions"
,"conclusions/relevance" :"conclusions"
,"conclusions/significance" :"conclusions"
,"conclusions/significances" :"conclusions"
,"conclusions/summary" :"conclusions"
,"condensation" :"conclusions"
,"condensed abstract" :"conclusions"
,"condition" :"background"
,"conditions" :"conclusions"
,"conference process" :"methods"
,"conflict of interest" :"background"
,"conflict-of-interest statement" :"background"
,"consensus" :"methods"
,"consensus position" :"methods"
,"consensus process" :"methods"
,"consensus statement" :"methods"
,"consequence" :"conclusions"
,"consequences" :"conclusions"
,"consideration" :"conclusions"
,"considerations" :"conclusions"
,"contact" :"background"
,"contacts" :"background"
,"content" :"background"
,"content analysis of literature" :"methods"
,"contents" :"background"
,"context" :"background"
,"context & background" :"background"
,"context & objective" :"objective"
,"context and aim" :"objective"
,"context and aims" :"objective"
,"context and objective" :"objective"
,"context and objectives" :"objective"
,"context and purpose" :"objective"
,"context and rationale" :"background"
,"context of case" :"background"
,"context of the problem" :"objective"
,"context/background" :"background"
,"context/objective" :"objective"
,"context/objectives" :"objective"
,"contextual issues" :"conclusions"
,"contraindication" :"methods"
,"contraindications" :"methods"
,"control" :"methods"
,"control data" :"methods"
,"control group" :"methods"
,"controls" :"methods"
,"controversial issues" :"objective"
,"costs" :"results"
,"count" :"methods"
,"course" :"results"
,"course and treatment" :"results"
,"credibility" :"methods"
,"criteria" :"methods"
,"critical issues" :"results"
,"critical issues and future directions" :"conclusions"
,"critique" :"results"
,"current & future development" :"conclusions"
,"current data" :"methods"
,"current knowledge" :"background"
,"current knowledge and key points" :"background"
,"current situation" :"background"
,"current situation and salient points" :"background"
,"current status" :"results"
,"data" :"methods"
,"data & methods" :"methods"
,"data abstraction" :"methods"
,"data acquisition" :"methods"
,"data analyses" :"methods"
,"data analysis" :"methods"
,"data analysis method" :"methods"
,"data and method" :"methods"
,"data and methods" :"methods"
,"data and sources" :"methods"
,"data capture" :"methods"
,"data collected" :"methods"
,"data collection" :"methods"
,"data collection & analysis" :"methods"
,"data collection and analysis" :"methods"
,"data collection method" :"methods"
,"data collection methods" :"methods"
,"data collection/analysis" :"methods"
,"data collection/extraction" :"methods"
,"data collection/extraction method" :"methods"
,"data collection/extraction methods" :"methods"
,"data extraction" :"methods"
,"data extraction and analysis" :"methods"
,"data extraction and data synthesis" :"methods"
,"data extraction and synthesis" :"methods"
,"data extraction methods" :"methods"
,"data extraction/synthesis" :"methods"
,"data identification" :"methods"
,"data identification and selection" :"methods"
,"data quality" :"methods"
,"data resources" :"methods"
,"data retrieval" :"methods"
,"data selection" :"methods"
,"data selection and extraction" :"methods"
,"data set" :"methods"
,"data source" :"methods"
,"data source and methods" :"methods"
,"data source and selection" :"methods"
,"data source and study selection" :"methods"
,"data source/study design" :"methods"
,"data source/study setting" :"methods"
,"data sources" :"methods"
,"data sources & selection" :"methods"
,"data sources & study setting" :"methods"
,"data sources and data extraction" :"methods"
,"data sources and extraction" :"methods"
,"data sources and methods" :"methods"
,"data sources and review methods" :"methods"
,"data sources and selection" :"methods"
,"data sources and selection criteria" :"methods"
,"data sources and setting" :"methods"
,"data sources and study design" :"methods"
,"data sources and study selection" :"methods"
,"data sources and study setting" :"methods"
,"data sources and synthesis" :"methods"
,"data sources, extraction, and synthesis" :"methods"
,"data sources, study selection, and data extraction" :"methods"
,"data sources/data collection" :"methods"
,"data sources/setting" :"methods"
,"data sources/study design" :"methods"
,"data sources/study selection" :"methods"
,"data sources/study setting" :"methods"
,"data sources/study settings" :"methods"
,"data sources/synthesis" :"methods"
,"data summary" :"methods"
,"data syntheses" :"methods"
,"data synthesis" :"results"
,"data synthesis and conclusion" :"conclusions"
,"data synthesis and conclusions" :"conclusions"
,"data synthesis/methods" :"methods"
,"data/results" :"results"
,"database" :"methods"
,"databases" :"methods"
,"databases used" :"methods"
,"date sources" :"methods"
,"declaration of interest" :"background"
,"definition" :"background"
,"definition of the problem" :"objective"
,"definitions" :"background"
,"demographics" :"methods"
,"dependent measures" :"methods"
,"dependent variable" :"methods"
,"dependent variables" :"methods"
,"description" :"methods"
,"description of case" :"methods"
,"description of cases" :"methods"
,"description of course" :"methods"
,"description of instrumentation" :"methods"
,"description of policy practice" :"methods"
,"description of program" :"methods"
,"description of project" :"methods"
,"description of study" :"methods"
,"description of system" :"methods"
,"description of systems" :"methods"
,"description of technique" :"methods"
,"description of technology/therapy" :"methods"
,"description of the case" :"methods"
,"description of the project" :"methods"
,"description of the project/innovation" :"methods"
,"description of the study" :"methods"
,"description of the system" :"methods"
,"descriptions" :"methods"
,"descriptor" :"background"
,"descriptors" :"background"
,"design" :"methods"
,"design & definition" :"methods"
,"design & intervention" :"methods"
,"design & method" :"methods"
,"design & methodology" :"methods"
,"design & methods" :"methods"
,"design & participants" :"methods"
,"design & setting" :"methods"
,"design & subjects" :"methods"
,"design and analysis" :"methods"
,"design and data sources" :"methods"
,"design and intervention" :"methods"
,"design and interventions" :"methods"
,"design and main outcome measures" :"methods"
,"design and materials" :"methods"
,"design and measurement" :"methods"
,"design and measurements" :"methods"
,"design and measures" :"methods"
,"design and method" :"methods"
,"design and methodology" :"methods"
,"design and methods" :"methods"
,"design and objective" :"objective"
,"design and outcome measures" :"methods"
,"design and participants" :"methods"
,"design and patient" :"methods"
,"design and patients" :"methods"
,"design and population" :"methods"
,"design and procedure" :"methods"
,"design and procedures" :"methods"
,"design and results" :"results"
,"design and sample" :"methods"
,"design and scope" :"methods"
,"design and setting" :"methods"
,"design and settings" :"methods"
,"design and study participants" :"methods"
,"design and study population" :"methods"
,"design and study sample" :"methods"
,"design and study subjects" :"methods"
,"design and subjects" :"methods"
,"design and volunteers" :"methods"
,"design classification" :"methods"
,"design methods" :"methods"
,"design of study" :"methods"
,"design of the study" :"methods"
,"design setting" :"methods"
,"design setting and participants" :"methods"
,"design setting and subjects" :"methods"
,"design study" :"methods"
,"design(s)" :"methods"
,"design, interventions, and main outcome measures" :"methods"
,"design, material and methods" :"methods"
,"design, materials & methods" :"methods"
,"design, materials and methods" :"methods"
,"design, materials, and methods" :"methods"
,"design, participants" :"methods"
,"design, participants and measurements" :"methods"
,"design, participants and setting" :"methods"
,"design, participants, & measurements" :"methods"
,"design, participants, and intervention" :"methods"
,"design, participants, and interventions" :"methods"
,"design, participants, and measures" :"methods"
,"design, participants, and methods" :"methods"
,"design, participants, and setting" :"methods"
,"design, participants, measurements" :"methods"
,"design, patients" :"methods"
,"design, patients and measurements" :"methods"
,"design, patients and methods" :"methods"
,"design, patients and setting" :"methods"
,"design, patients, & setting" :"methods"
,"design, patients, and interventions" :"methods"
,"design, patients, and main outcome measures" :"methods"
,"design, patients, and methods" :"methods"
,"design, patients, and setting" :"methods"
,"design, patients, measurements" :"methods"
,"design, setting" :"methods"
,"design, setting & participants" :"methods"
,"design, setting & patients" :"methods"
,"design, setting and methods" :"methods"
,"design, setting and participants" :"methods"
,"design, setting and patients" :"methods"
,"design, setting and subjects" :"methods"
,"design, setting participants, & measurements" :"methods"
,"design, setting, and methods" :"methods"
,"design, setting, and participants" :"methods"
,"design, setting, and patient" :"methods"
,"design, setting, and patients" :"methods"
,"design, setting, and population" :"methods"
,"design, setting, and subjects" :"methods"
,"design, setting, participants" :"methods"
,"design, setting, participants & measurements" :"methods"
,"design, setting, participants and intervention" :"methods"
,"design, setting, participants and interventions" :"methods"
,"design, setting, participants and measurements" :"methods"
,"design, setting, participants, & measurements" :"methods"
,"design, setting, participants, & methods" :"methods"
,"design, setting, participants, & objectives" :"methods"
,"design, setting, participants, and intervention" :"methods"
,"design, setting, participants, and interventions" :"methods"
,"design, setting, participants, and main outcome measures" :"methods"
,"design, setting, participants, and measurements" :"methods"
,"design, setting, participants, and measures" :"methods"
,"design, setting, participants, measurements" :"methods"
,"design, setting, patients" :"methods"
,"design, setting, patients, and intervention" :"methods"
,"design, setting, patients, and main outcome measure" :"methods"
,"design, setting, patients, interventions" :"methods"
,"design, setting, subjects" :"methods"
,"design, settings and participants" :"methods"
,"design, settings, and participants" :"methods"
,"design, settings, and patients" :"methods"
,"design, settings, and subjects" :"methods"
,"design, settings, participants, & measurements" :"methods"
,"design, settings, participants, & methods" :"methods"
,"design, subjects and intervention" :"methods"
,"design, subjects and measurements" :"methods"
,"design, subjects and methods" :"methods"
,"design, subjects and setting" :"methods"
,"design, subjects, and intervention" :"methods"
,"design, subjects, and setting" :"methods"
,"design/intervention" :"methods"
,"design/interventions" :"methods"
,"design/measurements" :"methods"
,"design/method" :"methods"
,"design/method/approach" :"methods"
,"design/methodology" :"methods"
,"design/methodology/approach" :"methods"
,"design/methods" :"methods"
,"design/outcome measures" :"methods"
,"design/participants" :"methods"
,"design/participants/setting" :"methods"
,"design/patients" :"methods"
,"design/patients/measurements" :"methods"
,"design/sample" :"methods"
,"design/setting" :"methods"
,"design/setting/participants" :"methods"
,"design/setting/participants/measurements" :"methods"
,"design/setting/patients" :"methods"
,"design/setting/subjects" :"methods"
,"design/subjects" :"methods"
,"designs" :"methods"
,"designs and methods" :"methods"
,"developer and funding" :"background"
,"developing recommendations" :"methods"
,"development" :"methods"
,"development and conclusion" :"conclusions"
,"development and conclusions" :"conclusions"
,"developments" :"conclusions"
,"devices" :"methods"
,"diabetes strategy evidence platform" :"methods"
,"diagnosis" :"methods"
,"diagnosis and course" :"methods"
,"diagnosis and management" :"methods"
,"diagnosis and therapy" :"methods"
,"diagnosis and treatment" :"methods"
,"diagnosis, therapy and clinical course" :"methods"
,"diagnosis, therapy and course" :"methods"
,"diagnosis, treatment and clinical course" :"methods"
,"diagnosis, treatment and course" :"methods"
,"diagnostic" :"methods"
,"diagnostic methods" :"methods"
,"diagnostic procedures" :"methods"
,"diagnostic test" :"methods"
,"diagnostic tests" :"methods"
,"diagnostic work-up" :"methods"
,"diagnostics" :"methods"
,"differential diagnosis" :"methods"
,"disclaimer" :"conclusions"
,"disclosure" :"background"
,"discussion" :"conclusions"
,"discussion & conclusion" :"conclusions"
,"discussion & summary" :"conclusions"
,"discussion - conclusion" :"conclusions"
,"discussion and conclusion" :"conclusions"
,"discussion and conclusions" :"conclusions"
,"discussion and evaluation" :"results"
,"discussion and implications" :"conclusions"
,"discussion and implications for practice" :"conclusions"
,"discussion and limitations" :"conclusions"
,"discussion and recommendations" :"conclusions"
,"discussion and results" :"results"
,"discussion and summary" :"conclusions"
,"discussion, conclusion" :"conclusions"
,"discussion-conclusion" :"conclusions"
,"discussion/conclusion" :"conclusions"
,"discussion/conclusions" :"conclusions"
,"discussions" :"conclusions"
,"discussions and conclusion" :"conclusions"
,"discussions and conclusions" :"conclusions"
,"discussions/conclusions" :"conclusions"
,"disease control" :"results"
,"disease management" :"methods"
,"disease overview" :"background"
,"disease signs" :"methods"
,"disease symptoms" :"methods"
,"dissemination" :"results"
,"donors and methods" :"methods"
,"duration" :"methods"
,"ebm rating" :"methods"
,"economic analysis" :"methods"
,"educational objective" :"objective"
,"educational objectives" :"objective"
,"effectiveness" :"results"
,"effects of change" :"results"
,"efficacy" :"results"
,"electronic supplementary material" :"background"
,"eligibility criteria" :"methods"
,"eligibility criteria for included studies" :"methods"
,"eligibility criteria for selecting studies" :"methods"
,"emerging areas for developing research" :"conclusions"
,"emplacement" :"methods"
,"end point" :"methods"
,"end points" :"methods"
,"endorsement" :"background"
,"endpoint" :"methods"
,"endpoints" :"methods"
,"enhanced version" :"background"
,"environment" :"methods"
,"epidemiology" :"background"
,"epilogue" :"conclusions"
,"equipment" :"methods"
,"equipment and methods" :"methods"
,"essential results" :"results"
,"ethical issues" :"methods"
,"ethical issues and approval" :"methods"
,"ethics" :"methods"
,"ethics and dissemination" :"background"
,"ethno pharmacological relevance" :"background"
,"ethnopharmacological evidence" :"background"
,"ethnopharmacological importance" :"background"
,"ethnopharmacological relevance" :"background"
,"ethnopharmacological significance" :"background"
,"ethnopharmacology" :"background"
,"ethnopharmacology relevance" :"background"
,"etiology" :"background"
,"evaluation" :"results"
,"evaluation aims" :"objective"
,"evaluation design" :"methods"
,"evaluation method" :"methods"
,"evaluation methods" :"methods"
,"evaluation results" :"results"
,"evaluations" :"methods"
,"evaluations/measurements" :"methods"
,"evidence" :"methods"
,"evidence acquisition" :"methods"
,"evidence acquisition and synthesis" :"methods"
,"evidence acquisitions" :"methods"
,"evidence and consensus process" :"methods"
,"evidence and information sources" :"methods"
,"evidence base" :"methods"
,"evidence level" :"methods"
,"evidence review" :"methods"
,"evidence summary" :"methods"
,"evidence synthesis" :"results"
,"evidence-based analysis methods" :"methods"
,"evolution" :"results"
,"examination" :"results"
,"examinations" :"results"
,"examinees and methods" :"methods"
,"examples" :"results"
,"exclusion criteria" :"methods"
,"exclusions" :"methods"
,"exegese" :"methods"
,"exegesis" :"methods"
,"expected outcomes" :"objective"
,"expected results" :"results"
,"experience" :"results"
,"experience and results" :"results"
,"experiences" :"results"
,"experiment" :"methods"
,"experiment design" :"methods"
,"experimental" :"methods"
,"experimental animals" :"methods"
,"experimental approach" :"methods"
,"experimental approach & key results" :"results"
,"experimental approaches" :"methods"
,"experimental design" :"methods"
,"experimental design and results" :"results"
,"experimental designs" :"methods"
,"experimental intervention" :"methods"
,"experimental interventions" :"methods"
,"experimental material" :"methods"
,"experimental materials" :"methods"
,"experimental methods" :"methods"
,"experimental objectives" :"objective"
,"experimental procedure" :"methods"
,"experimental procedures" :"methods"
,"experimental protocol" :"methods"
,"experimental studies" :"methods"
,"experimental subjects" :"methods"
,"experimental variable" :"methods"
,"experimental variables" :"methods"
,"experiments" :"methods"
,"experiments and results" :"results"
,"expert opinion" :"conclusions"
,"exposure" :"methods"
,"exposure measures" :"methods"
,"exposures" :"methods"
,"extraction" :"methods"
,"extraction methods" :"methods"
,"facility" :"methods"
,"factor" :"methods"
,"factors" :"methods"
,"feasibility" :"results"
,"features" :"methods"
,"final considerations" :"conclusions"
,"final diagnosis" :"methods"
,"final remarks" :"conclusions"
,"financial disclosure" :"background"
,"financial disclosure(s)" :"background"
,"financial disclosures" :"background"
,"finding" :"results"
,"findings" :"results"
,"findings and conclusion" :"conclusions"
,"findings and conclusions" :"conclusions"
,"findings and discussion" :"conclusions"
,"findings and implications" :"conclusions"
,"findings and outcomes" :"results"
,"findings and practice implications" :"conclusions"
,"findings and recommendations" :"results"
,"findings/conclusion" :"conclusions"
,"findings/conclusions" :"conclusions"
,"findings/results" :"results"
,"first case" :"methods"
,"focus" :"objective"
,"focused question" :"objective"
,"follow up" :"results"
,"follow-up" :"results"
,"format" :"methods"
,"foundation" :"background"
,"framework" :"methods"
,"fundamentals" :"objective"
,"funding" :"background"
,"funding source" :"background"
,"funding sources" :"background"
,"future" :"conclusions"
,"future and projects" :"conclusions"
,"future directions" :"conclusions"
,"future perspectives" :"conclusions"
,"future prospect and projects" :"conclusions"
,"future prospects" :"conclusions"
,"future prospects and projects" :"conclusions"
,"future research" :"conclusions"
,"future work" :"conclusions"
,"general methods" :"methods"
,"general question" :"objective"
,"general significance" :"conclusions"
,"generalizability to other populations" :"conclusions"
,"genetic toxicology" :"results"
,"genetics" :"background"
,"global importance" :"background"
,"goal" :"objective"
,"goal and methods" :"objective"
,"goal of study" :"objective"
,"goal of surgery" :"objective"
,"goal of the study" :"objective"
,"goal of this study" :"objective"
,"goal of work" :"objective"
,"goal, scope and background" :"objective"
,"goal, scope, and background" :"objective"
,"goals" :"objective"
,"goals and background" :"objective"
,"goals and objectives" :"objective"
,"goals of the study" :"objective"
,"goals of the work" :"objective"
,"goals of this study" :"objective"
,"goals of work" :"objective"
,"goals, scope and background" :"objective"
,"goals/background" :"objective"
,"group and methods" :"methods"
,"group of patients and methods" :"methods"
,"growing points" :"conclusions"
,"growing points and areas timely for developing research" :"conclusions"
,"guideline question" :"objective"
,"guideline questions" :"objective"
,"guidelines" :"conclusions"
,"harms" :"results"
,"health political background" :"background"
,"highlights" :"conclusions"
,"histology" :"results"
,"histopathology" :"results"
,"historical aspects" :"background"
,"historical background" :"background"
,"historical overview" :"background"
,"historical perspective" :"background"
,"history" :"methods"
,"history and admission diagnosis" :"background"
,"history and admission findings" :"methods"
,"history and clinical data" :"methods"
,"history and clinical finding" :"methods"
,"history and clinical findings" :"methods"
,"history and clinical presentation" :"methods"
,"history and clinically findings" :"methods"
,"history and examination" :"methods"
,"history and findings" :"methods"
,"history and findings on admission" :"methods"
,"history and general investigations" :"methods"
,"history and physical examination" :"methods"
,"history and physical findings" :"methods"
,"history and presenting complaint" :"methods"
,"history and reason for admission" :"methods"
,"history and signs" :"methods"
,"host range" :"methods"
,"human data synthesis" :"results"
,"hypertension" :"results"
,"hypotheses" :"objective"
,"hypothesis" :"objective"
,"hypothesis and aims" :"objective"
,"hypothesis and background" :"objective"
,"hypothesis and objectives" :"objective"
,"hypothesis/background" :"objective"
,"hypothesis/objective" :"objective"
,"hypothesis/objectives" :"objective"
,"hypothesis/problem" :"objective"
,"hypothesis/purpose" :"objective"
,"identification" :"methods"
,"illustrative case" :"methods"
,"illustrative cases" :"methods"
,"imaging" :"methods"
,"impact" :"conclusions"
,"impact for human medicine" :"conclusions"
,"impact of the study" :"conclusions"
,"impact on industry" :"conclusions"
,"impact on research, practice, and policy" :"conclusions"
,"impact on the industry" :"conclusions"
,"impact on traffic safety" :"conclusions"
,"impacts" :"conclusions"
,"implementation" :"methods"
,"implication" :"conclusions"
,"implication for cancer survivors" :"conclusions"
,"implication for further research" :"conclusions"
,"implication for nursing management" :"conclusions"
,"implication for nursing practice" :"conclusions"
,"implication for practice" :"conclusions"
,"implication of the hypothesis" :"conclusions"
,"implication statement" :"conclusions"
,"implications" :"conclusions"
,"implications and action" :"conclusions"
,"implications and conclusions" :"conclusions"
,"implications for cancer survivors" :"conclusions"
,"implications for case management" :"conclusions"
,"implications for case management practice" :"conclusions"
,"implications for clinical practice" :"conclusions"
,"implications for cm practice" :"conclusions"
,"implications for further research" :"conclusions"
,"implications for future research" :"conclusions"
,"implications for health care provision" :"conclusions"
,"implications for health care provision and use" :"conclusions"
,"implications for health policies" :"conclusions"
,"implications for health policy" :"conclusions"
,"implications for nurse managers" :"conclusions"
,"implications for nursing" :"conclusions"
,"implications for nursing and health policy" :"conclusions"
,"implications for nursing management" :"conclusions"
,"implications for nursing practice" :"conclusions"
,"implications for nursing research" :"conclusions"
,"implications for policy" :"conclusions"
,"implications for practice" :"conclusions"
,"implications for practice/research" :"conclusions"
,"implications for practise" :"conclusions"
,"implications for public health" :"conclusions"
,"implications for public health practice" :"conclusions"
,"implications for rehabilitation" :"conclusions"
,"implications for research" :"conclusions"
,"implications for research and practice" :"conclusions"
,"implications for research/practice" :"conclusions"
,"implications of the hypothesis" :"conclusions"
,"implications statement" :"conclusions"
,"implications/conclusions" :"conclusions"
,"importance" :"objective"
,"importance of the field" :"background"
,"importance to the field" :"background"
,"important findings" :"results"
,"in clinical practice" :"conclusions"
,"in conclusion" :"conclusions"
,"in practice" :"conclusions"
,"in summary" :"conclusions"
,"in vitro studies" :"methods"
,"in vivo studies" :"methods"
,"incidence" :"background"
,"included studies" :"methods"
,"inclusion" :"methods"
,"inclusion & exclusion criteria" :"methods"
,"inclusion & exclusions" :"methods"
,"inclusion and exclusion criteria" :"methods"
,"inclusion criteria" :"methods"
,"independent variable" :"methods"
,"independent variables" :"methods"
,"index test" :"methods"
,"index tests" :"methods"
,"indication" :"methods"
,"indications" :"methods"
,"individuals" :"methods"
,"individuals and methods" :"methods"
,"infants and methods" :"methods"
,"infection" :"background"
,"inference" :"conclusions"
,"information sources" :"methods"
,"initial assessment" :"methods"
,"injury patterns" :"methods"
,"innovation" :"methods"
,"innovation and conclusion" :"conclusions"
,"innovation and conclusions" :"conclusions"
,"innovation and implications" :"conclusions"
,"innovations" :"results"
,"institution" :"methods"
,"instruction" :"background"
,"instrument" :"methods"
,"instrumentation" :"methods"
,"instruments" :"methods"
,"instruments and methods" :"methods"
,"integrative significance" :"conclusions"
,"intention" :"objective"
,"intention of the study" :"objective"
,"intention, goal, scope, background" :"background"
,"interpretation" :"conclusions"
,"interpretation & conclusion" :"conclusions"
,"interpretation & conclusions" :"conclusions"
,"interpretation and conclusion" :"conclusions"
,"interpretation and conclusions" :"conclusions"
,"interpretation/conclusion" :"conclusions"
,"interpretation/conclusions" :"conclusions"
,"interpretations" :"conclusions"
,"interpretations & conclusion" :"conclusions"
,"interpretations & conclusions" :"conclusions"
,"interpretations and conclusions" :"conclusions"
,"intervention" :"methods"
,"intervention & measurements" :"methods"
,"intervention and main outcome measure" :"methods"
,"intervention and main outcome measures" :"methods"
,"intervention and measurements" :"methods"
,"intervention and methods" :"methods"
,"intervention and outcome" :"results"
,"intervention and outcome measures" :"methods"
,"intervention and outcomes" :"results"
,"intervention and results" :"results"
,"intervention and technique" :"methods"
,"intervention and testing" :"methods"
,"intervention(s)" :"methods"
,"intervention(s) and main outcome measure(s)" :"methods"
,"intervention/methods" :"methods"
,"intervention/technique" :"methods"
,"interventions" :"methods"
,"interventions and main outcome measurements" :"methods"
,"interventions and main outcome measures" :"methods"
,"interventions and main results" :"results"
,"interventions and measurements" :"methods"
,"interventions and methods" :"methods"
,"interventions and outcome" :"results"
,"interventions and outcome measures" :"methods"
,"interventions and outcomes" :"results"
,"interventions and results" :"results"
,"interventions(s)" :"methods"
,"interventions, measurements, and main results" :"results"
,"interventions/methods" :"methods"
,"intro" :"background"
,"introduction" :"background"
,"introduction & aim" :"objective"
,"introduction & background" :"background"
,"introduction & objective" :"objective"
,"introduction & objectives" :"objective"
,"introduction & purpose" :"objective"
,"introduction and aim" :"objective"
,"introduction and aim of study" :"objective"
,"introduction and aim of the study" :"objective"
,"introduction and aims" :"objective"
,"introduction and aims of the study" :"objective"
,"introduction and background" :"background"
,"introduction and clinical case" :"methods"
,"introduction and clinical cases" :"methods"
,"introduction and design" :"objective"
,"introduction and development" :"background"
,"introduction and goal" :"objective"
,"introduction and goals" :"objective"
,"introduction and hypothesis" :"objective"
,"introduction and material" :"background"
,"introduction and method" :"methods"
,"introduction and methodology" :"methods"
,"introduction and methods" :"methods"
,"introduction and object" :"objective"
,"introduction and objective" :"objective"
,"introduction and objectives" :"objective"
,"introduction and potentials of classical radiotherapy" :"objective"
,"introduction and problem" :"objective"
,"introduction and prognosis" :"objective"
,"introduction and proposed study" :"objective"
,"introduction and purpose" :"objective"
,"introduction and rationale" :"objective"
,"introduction or background" :"background"
,"introduction/aim" :"objective"
,"introduction/aims" :"objective"
,"introduction/background" :"background"
,"introduction/hypothesis" :"objective"
,"introduction/methods" :"methods"
,"introduction/objective" :"objective"
,"introduction/objectives" :"objective"
,"introduction/purpose" :"objective"
,"introductions" :"background"
,"investigated group" :"methods"
,"investigation" :"methods"
,"investigation and diagnosis" :"methods"
,"investigation(s)" :"methods"
,"investigations" :"methods"
,"investigations and diagnosis" :"methods"
,"investigations and treatment" :"methods"
,"investigations, diagnosis and treatment" :"methods"
,"investigations, treatment and course" :"methods"
,"issue" :"objective"
,"issue addressed" :"objective"
,"issues" :"objective"
,"issues addressed" :"objective"
,"issues and purpose" :"objective"
,"jel classification" :"background"
,"jel codes" :"background"
,"justification" :"background"
,"key conclusion" :"conclusions"
,"key conclusion and implications for practice" :"conclusions"
,"key conclusions" :"conclusions"
,"key conclusions and implications" :"conclusions"
,"key conclusions and implications for practice" :"conclusions"
,"key exposure/study factor" :"methods"
,"key finding" :"results"
,"key findings" :"results"
,"key findings and implications" :"results"
,"key issue" :"results"
,"key issue(s)" :"results"
,"key issues" :"results"
,"key learning point" :"conclusions"
,"key learning points" :"conclusions"
,"key limitations" :"conclusions"
,"key measures" :"methods"
,"key measures for improvement" :"methods"
,"key message" :"conclusions"
,"key messages" :"conclusions"
,"key points" :"conclusions"
,"key practitioner message" :"conclusions"
,"key questions and answers" :"conclusions"
,"key recommendations" :"conclusions"
,"key results" :"results"
,"key results and conclusions" :"conclusions"
,"key risk/study factor" :"methods"
,"key study factor" :"methods"
,"key words" :"background"
,"keys to success" :"conclusions"
,"keywords" :"background"
,"knowledge translation" :"conclusions"
,"laboratory findings" :"results"
,"laboratory tests" :"methods"
,"learning objective" :"objective"
,"learning objectives" :"objective"
,"learning outcomes" :"objective"
,"learning points" :"conclusions"
,"lessons" :"conclusions"
,"lessons and messages" :"conclusions"
,"lessons learned" :"conclusions"
,"lessons learnt" :"conclusions"
,"level iii" :"methods"
,"level of clinical evidence" :"methods"
,"level of evidence" :"methods"
,"level of evidence i" :"methods"
,"level of evidence ii" :"methods"
,"level of evidence iii" :"methods"
,"level of evidence iv" :"methods"
,"level of evidence v" :"methods"
,"level of proof" :"methods"
,"levels of evidence" :"methods"
,"limitation" :"conclusions"
,"limitation, reasons for caution" :"conclusions"
,"limitations" :"conclusions"
,"limitations and conclusions" :"conclusions"
,"limitations and reasons for caution" :"conclusions"
,"limitations of the study" :"conclusions"
,"limitations, reason for caution" :"conclusions"
,"limitations, reasons for caution" :"conclusions"
,"limitations, reasons for cautions" :"conclusions"
,"limits" :"conclusions"
,"linked article" :"background"
,"linked articles" :"background"
,"linking evidence to action" :"conclusions"
,"literature" :"methods"
,"literature findings" :"results"
,"literature review" :"methods"
,"literature reviewed" :"methods"
,"literature search" :"methods"
,"literature survey" :"methods"
,"local setting" :"methods"
,"location" :"methods"
,"locations" :"methods"
,"main components of program" :"methods"
,"main conclusion" :"conclusions"
,"main conclusions" :"conclusions"
,"main contribution" :"results"
,"main exposure" :"methods"
,"main exposure measure" :"methods"
,"main exposure measures" :"methods"
,"main exposures" :"methods"
,"main features" :"methods"
,"main finding" :"results"
,"main findings" :"results"
,"main independent variables" :"methods"
,"main issues" :"results"
,"main measure" :"methods"
,"main measurement" :"methods"
,"main measurements" :"methods"
,"main measurements and results" :"results"
,"main measures" :"methods"
,"main measures and results" :"results"
,"main message" :"results"
,"main messages" :"conclusions"
,"main method" :"methods"
,"main methods" :"methods"
,"main methods and key findings" :"results"
,"main objective" :"objective"
,"main objectives" :"objective"
,"main observation" :"methods"
,"main observations" :"methods"
,"main observations and results" :"methods"
,"main outcome" :"results"
,"main outcome and measure" :"methods"
,"main outcome and measurements" :"methods"
,"main outcome and measures" :"methods"
,"main outcome and results" :"results"
,"main outcome criteria" :"methods"
,"main outcome findings" :"results"
,"main outcome measure" :"methods"
,"main outcome measure and results" :"results"
,"main outcome measure(s)" :"methods"
,"main outcome measured" :"methods"
,"main outcome measurement" :"methods"
,"main outcome measurement(s)" :"methods"
,"main outcome measurements" :"methods"
,"main outcome measurements and results" :"results"
,"main outcome measures" :"methods"
,"main outcome measures and methods" :"methods"
,"main outcome measures and results" :"results"
,"main outcome measures(s)" :"methods"
,"main outcome measures/results" :"results"
,"main outcome methods" :"methods"
,"main outcome parameters" :"methods"
,"main outcome results" :"results"
,"main outcome variable" :"methods"
,"main outcome variables" :"methods"
,"main outcomes" :"results"
,"main outcomes and measure" :"methods"
,"main outcomes and measures" :"methods"
,"main outcomes and results" :"results"
,"main outcomes measure" :"methods"
,"main outcomes measure(s)" :"methods"
,"main outcomes measured" :"methods"
,"main outcomes measurements" :"methods"
,"main outcomes measures" :"methods"
,"main points" :"results"
,"main problem" :"objective"
,"main purpose" :"objective"
,"main recommendations" :"conclusions"
,"main research classifications" :"methods"
,"main research variable" :"methods"
,"main research variables" :"methods"
,"main result" :"results"
,"main results" :"results"
,"main results and conclusions" :"conclusions"
,"main results and role of chance" :"results"
,"main results and the role of chance" :"results"
,"main study measures" :"methods"
,"main topics" :"objective"
,"main variables" :"methods"
,"main variables examined" :"methods"
,"main variables of interest" :"methods"
,"main variables studied" :"methods"
,"major conclusion" :"conclusions"
,"major conclusions" :"conclusions"
,"major findings" :"results"
,"major outcome measures" :"methods"
,"major points" :"results"
,"major results" :"results"
,"management" :"results"
,"management of refractory disease" :"conclusions"
,"material" :"methods"
,"material & method" :"methods"
,"material & methods" :"methods"
,"material and discussion" :"results"
,"material and method" :"methods"
,"material and methodology" :"methods"
,"material and methods" :"methods"
,"material and patients" :"methods"
,"material and results" :"results"
,"material and subjects" :"methods"
,"material and surgical technique" :"methods"
,"material and treatment" :"methods"
,"material method" :"methods"
,"material methods" :"methods"
,"material of study" :"methods"
,"material or subjects" :"methods"
,"material, methods" :"methods"
,"material, methods and results" :"results"
,"material, patients and methods" :"methods"
,"material-method" :"methods"
,"material-methods" :"methods"
,"material/method" :"methods"
,"material/methods" :"methods"
,"material/patients & methods" :"methods"
,"material/subjects and methods" :"methods"
,"materials" :"methods"
,"materials & method" :"methods"
,"materials & methodology" :"methods"
,"materials & methods" :"methods"
,"materials & results" :"results"
,"materials and  methods" :"methods"
,"materials and interventions" :"methods"
,"materials and materials" :"methods"
,"materials and method" :"methods"
,"materials and methodology" :"methods"
,"materials and methods" :"methods"
,"materials and methods and results" :"results"
,"materials and methods/results" :"results"
,"materials and patients" :"methods"
,"materials and results" :"results"
,"materials and subjects" :"methods"
,"materials and surgical technique" :"methods"
,"materials and treatment" :"methods"
,"materials methods" :"methods"
,"materials of study" :"methods"
,"materials or subjects" :"methods"
,"materials, methods" :"methods"
,"materials, methods and results" :"results"
,"materials, methods, and results" :"results"
,"materials, setting and methods" :"methods"
,"materials, setting, methods" :"methods"
,"materials-methods" :"methods"
,"materials/method" :"methods"
,"materials/methods" :"methods"
,"materials/patients and methods" :"methods"
,"materials/subjects and methods" :"methods"
,"mean outcome measure" :"methods"
,"mean outcome measure(s)" :"methods"
,"mean outcome measures" :"methods"
,"measure" :"methods"
,"measurement" :"methods"
,"measurement & outcomes" :"methods"
,"measurement & results" :"results"
,"measurement and findings" :"results"
,"measurement and main result" :"results"
,"measurement and main results" :"results"
,"measurement and results" :"results"
,"measurements" :"methods"
,"measurements & outcomes" :"methods"
,"measurements & results" :"results"
,"measurements and analysis" :"methods"
,"measurements and findings" :"results"
,"measurements and interventions" :"methods"
,"measurements and main findings" :"results"
,"measurements and main outcomes" :"results"
,"measurements and main result" :"results"
,"measurements and main results" :"results"
,"measurements and methods" :"methods"
,"measurements and outcomes" :"methods"
,"measurements and results" :"results"
,"measurements/main results" :"results"
,"measurements/results" :"results"
,"measures" :"methods"
,"measures and analysis" :"methods"
,"measures and main results" :"results"
,"measures and results" :"results"
,"measures of outcome" :"methods"
,"measurments" :"methods"
,"mechanism of action" :"results"
,"mechanisms" :"results"
,"mechanisms of action" :"results"
,"medical history" :"methods"
,"medical treatment" :"methods"
,"medication" :"methods"
,"message" :"conclusions"
,"method" :"methods"
,"method & material" :"methods"
,"method & materials" :"methods"
,"method & procedures" :"methods"
,"method & results" :"results"
,"method and analysis" :"methods"
,"method and clinical material" :"methods"
,"method and design" :"methods"
,"method and findings" :"results"
,"method and material" :"methods"
,"method and materials" :"methods"
,"method and participants" :"methods"
,"method and patients" :"methods"
,"method and procedure" :"methods"
,"method and procedures" :"methods"
,"method and result" :"results"
,"method and results" :"results"
,"method and sample" :"methods"
,"method and subjects" :"methods"
,"method of approach" :"methods"
,"method of study" :"methods"
,"method of study selection" :"methods"
,"method summary" :"methods"
,"method(s)" :"methods"
,"method/design" :"methods"
,"method/materials" :"methods"
,"method/patients" :"methods"
,"method/principal findings" :"results"
,"method/result" :"results"
,"method/results" :"results"
,"methodical innovations" :"methods"
,"methodologic approach" :"methods"
,"methodological approach" :"methods"
,"methodological design" :"methods"
,"methodological design and justification" :"methods"
,"methodological procedures" :"methods"
,"methodological quality" :"methods"
,"methodologies" :"methods"
,"methodologies/principal findings" :"results"
,"methodology" :"methods"
,"methodology & principal findings" :"results"
,"methodology & principle findings" :"results"
,"methodology and findings" :"results"
,"methodology and patients" :"methods"
,"methodology and principal finding" :"results"
,"methodology and principal findings" :"results"
,"methodology and principle findings" :"results"
,"methodology and results" :"results"
,"methodology and sample" :"methods"
,"methodology principal findings" :"results"
,"methodology/ principal findings" :"results"
,"methodology/approach" :"methods"
,"methodology/findings" :"results"
,"methodology/main findings" :"results"
,"methodology/main results" :"results"
,"methodology/principal" :"results"
,"methodology/principal finding" :"results"
,"methodology/principal findings" :"results"
,"methodology/principle findings" :"results"
,"methodology/results" :"results"
,"methods" :"methods"
,"methods & design" :"methods"
,"methods & findings" :"results"
,"methods & material" :"methods"
,"methods & materials" :"methods"
,"methods & outcome measures" :"methods"
,"methods & patients" :"methods"
,"methods & procedure" :"methods"
,"methods & procedures" :"methods"
,"methods & procedures/outcomes & results" :"results"
,"methods & results" :"results"
,"methods & study design" :"methods"
,"methods - data sources" :"methods"
,"methods - study selection" :"methods"
,"methods / design" :"methods"
,"methods and aims" :"objective"
,"methods and analyses" :"methods"
,"methods and analysis" :"methods"
,"methods and approach" :"methods"
,"methods and conclusions" :"conclusions"
,"methods and data" :"methods"
,"methods and design" :"methods"
,"methods and discussion" :"results"
,"methods and finding" :"results"
,"methods and findings" :"results"
,"methods and focus" :"objective"
,"methods and interventions" :"methods"
,"methods and key results" :"results"
,"methods and main outcome measures" :"methods"
,"methods and main outcomes" :"methods"
,"methods and main results" :"results"
,"methods and material" :"methods"
,"methods and materials" :"methods"
,"methods and measurements" :"methods"
,"methods and measures" :"methods"
,"methods and methods" :"methods"
,"methods and objectives" :"objective"
,"methods and outcome measures" :"methods"
,"methods and outcomes" :"methods"
,"methods and participants" :"methods"
,"methods and patients" :"methods"
,"methods and population" :"methods"
,"methods and principal findings" :"results"
,"methods and procedure" :"methods"
,"methods and procedures" :"methods"
,"methods and result" :"results"
,"methods and results" :"results"
,"methods and sample" :"methods"
,"methods and setting" :"methods"
,"methods and study design" :"methods"
,"methods and subjects" :"methods"
,"methods and technique" :"methods"
,"methods design" :"methods"
,"methods of analysis" :"methods"
,"methods of study" :"methods"
,"methods of study selection" :"methods"
,"methods used" :"methods"
,"methods, results" :"results"
,"methods, results and conclusions" :"conclusions"
,"methods-results" :"results"
,"methods/description" :"methods"
,"methods/design" :"methods"
,"methods/designs" :"methods"
,"methods/findings" :"results"
,"methods/literature reviewed" :"methods"
,"methods/materials" :"methods"
,"methods/patients" :"methods"
,"methods/principal finding" :"results"
,"methods/principal findings" :"results"
,"methods/results" :"results"
,"methods/sample" :"methods"
,"methods/setting" :"methods"
,"methods/study design" :"methods"
,"methods/subjects" :"methods"
,"methodsâ€„andâ€„results" :"results"
,"mini summary" :"background"
,"mini-abstract" :"conclusions"
,"model" :"methods"
,"model & outcomes" :"methods"
,"model description" :"methods"
,"model, perspective, & time frame" :"methods"
,"model, perspective, & timeframe" :"methods"
,"models" :"methods"
,"mortality" :"results"
,"motivation" :"background"
,"motivation and results" :"results"
,"motivations" :"background"
,"motivations and results" :"results"
,"new information" :"conclusions"
,"new method" :"methods"
,"new methods" :"methods"
,"new or unique information provided" :"conclusions"
,"next steps" :"conclusions"
,"no level assigned" :"methods"
,"null hypothesis" :"objective"
,"numbers" :"methods"
,"nursing implications" :"conclusions"
,"object" :"objective"
,"object & method" :"objective"
,"object and background" :"objective"
,"object and design" :"objective"
,"object and method" :"objective"
,"object and methods" :"objective"
,"object of study" :"objective"
,"object of the study" :"objective"
,"object of work" :"objective"
,"objections" :"objective"
,"objective" :"objective"
,"objective & aims" :"objective"
,"objective & design" :"objective"
,"objective & method" :"objective"
,"objective & methods" :"objective"
,"objective and aim" :"objective"
,"objective and background" :"objective"
,"objective and background data" :"objective"
,"objective and conclusion" :"conclusions"
,"objective and design" :"objective"
,"objective and hypothesis" :"objective"
,"objective and importance" :"objective"
,"objective and method" :"objective"
,"objective and methods" :"objective"
,"objective and motivation" :"objective"
,"objective and participants" :"objective"
,"objective and patients" :"objective"
,"objective and purpose" :"objective"
,"objective and rationale" :"objective"
,"objective and results" :"objective"
,"objective and setting" :"objective"
,"objective and study design" :"objective"
,"objective and subjects" :"objective"
,"objective and summary background data" :"objective"
,"objective of program" :"objective"
,"objective of review" :"objective"
,"objective of study" :"objective"
,"objective of the program" :"objective"
,"objective of the study" :"objective"
,"objective(s)" :"objective"
,"objective, design, and setting" :"objective"
,"objective/aim" :"objective"
,"objective/aims" :"objective"
,"objective/background" :"objective"
,"objective/design" :"objective"
,"objective/design/patients" :"objective"
,"objective/goal" :"objective"
,"objective/hypotheses" :"objective"
,"objective/hypothesis" :"objective"
,"objective/method" :"objective"
,"objective/methods" :"objective"
,"objective/patients" :"objective"
,"objective/purpose" :"objective"
,"objective/setting" :"objective"
,"objectives" :"objective"
,"objectives & methods" :"objective"
,"objectives and aim" :"objective"
,"objectives and aims" :"objective"
,"objectives and background" :"objective"
,"objectives and design" :"objective"
,"objectives and goal" :"objective"
,"objectives and hypothesis" :"objective"
,"objectives and method" :"objective"
,"objectives and methods" :"objective"
,"objectives and patients" :"objective"
,"objectives and rationale" :"objective"
,"objectives and results" :"objective"
,"objectives and setting" :"objective"
,"objectives and study design" :"objective"
,"objectives of study" :"objective"
,"objectives of the review" :"objective"
,"objectives of the study" :"objective"
,"objectives, patients and methods" :"objective"
,"objectives/ hypothesis" :"objective"
,"objectives/aim" :"objective"
,"objectives/aims" :"objective"
,"objectives/background" :"objective"
,"objectives/design" :"objective"
,"objectives/goal" :"objective"
,"objectives/hypotheses" :"objective"
,"objectives/hypothesis" :"objective"
,"objectives/methods" :"objective"
,"objectives/purpose" :"objective"
,"objectives/purposes" :"objective"
,"objectives/study design" :"objective"
,"objects" :"objective"
,"objects and methods" :"objective"
,"observation" :"methods"
,"observation & discussion" :"conclusions"
,"observation & results" :"results"
,"observation and results" :"results"
,"observation procedure" :"methods"
,"observation procedures" :"methods"
,"observational procedure" :"methods"
,"observations" :"methods"
,"observations and results" :"results"
,"ongoing issues" :"conclusions"
,"open peer review" :"background"
,"operations" :"methods"
,"operative procedure" :"methods"
,"operative technique" :"methods"
,"opportunities" :"methods"
,"options" :"methods"
,"options and outcomes" :"methods"
,"organization" :"methods"
,"organizing construct" :"methods"
,"organizing construct and methods" :"methods"
,"organizing constructs" :"background"
,"organizing framework" :"methods"
,"originality" :"conclusions"
,"originality/value" :"conclusions"
,"originality/value of chapter" :"conclusions"
,"origins of information" :"methods"
,"other measurements" :"methods"
,"other participants" :"methods"
,"outcome" :"results"
,"outcome & measurement" :"methods"
,"outcome & measurements" :"methods"
,"outcome & measures" :"methods"
,"outcome & results" :"results"
,"outcome and results" :"results"
,"outcome assessment" :"methods"
,"outcome measure" :"methods"
,"outcome measure(s)" :"methods"
,"outcome measured" :"methods"
,"outcome measurement" :"methods"
,"outcome measurements" :"methods"
,"outcome measurements and statistical analysis" :"methods"
,"outcome measures" :"methods"
,"outcome measures & results" :"results"
,"outcome measures and results" :"results"
,"outcome parameters" :"methods"
,"outcome variable" :"methods"
,"outcome variables" :"methods"
,"outcomes" :"results"
,"outcomes & measurement" :"methods"
,"outcomes & measurements" :"methods"
,"outcomes & measures" :"methods"
,"outcomes & other measurements" :"methods"
,"outcomes & results" :"results"
,"outcomes and measurements" :"methods"
,"outcomes and results" :"results"
,"outcomes assessment" :"methods"
,"outcomes measure" :"methods"
,"outcomes measured" :"methods"
,"outcomes measures" :"methods"
,"outcomes of interest" :"methods"
,"outcomes summary" :"conclusions"
,"outcomes/results" :"results"
,"outline" :"objective"
,"outline of cases" :"methods"
,"outlook" :"conclusions"
,"output" :"results"
,"outputs" :"results"
,"overall approach to quality and safety" :"methods"
,"overall article objective" :"objective"
,"overall article objectives" :"objective"
,"overall strength of evidence" :"methods"
,"overview" :"background"
,"overview of literature" :"background"
,"paper aim" :"objective"
,"parameters" :"methods"
,"participant" :"methods"
,"participant(s)" :"methods"
,"participants" :"methods"
,"participants & setting" :"methods"
,"participants & settings" :"methods"
,"participants and context" :"methods"
,"participants and controls" :"methods"
,"participants and design" :"methods"
,"participants and intervention" :"methods"
,"participants and interventions" :"methods"
,"participants and main outcome measures" :"methods"
,"participants and measurements" :"methods"
,"participants and measures" :"methods"
,"participants and method" :"methods"
,"participants and methods" :"methods"
,"participants and outcome measures" :"methods"
,"participants and patients" :"methods"
,"participants and setting" :"methods"
,"participants and settings" :"methods"
,"participants and/or controls" :"methods"
,"participants or samples" :"methods"
,"participants, design and setting" :"methods"
,"participants, design, and setting" :"methods"
,"participants, setting and methods" :"methods"
,"participants, setting, methods" :"methods"
,"participants/intervention" :"methods"
,"participants/interventions" :"methods"
,"participants/material, setting, methods" :"methods"
,"participants/materials, setting and methods" :"methods"
,"participants/materials, setting, and methods" :"methods"
,"participants/materials, setting, methods" :"methods"
,"participants/materials, settings, methods" :"methods"
,"participants/methods" :"methods"
,"participants/patients" :"methods"
,"participants/setting" :"methods"
,"participants/settings" :"methods"
,"participation" :"methods"
,"pathogenesis" :"methods"
,"pathological findings" :"results"
,"pathology" :"methods"
,"pathophysiology" :"methods"
,"patient" :"methods"
,"patient & method" :"methods"
,"patient & methods" :"methods"
,"patient and intervention" :"methods"
,"patient and method" :"methods"
,"patient and methods" :"methods"
,"patient and results" :"results"
,"patient case" :"methods"
,"patient characteristics" :"methods"
,"patient description" :"methods"
,"patient findings" :"methods"
,"patient group" :"methods"
,"patient history" :"methods"
,"patient material" :"methods"
,"patient population" :"methods"
,"patient population and methods" :"methods"
,"patient presentation" :"methods"
,"patient report" :"methods"
,"patient sample" :"methods"
,"patient sample and methodology" :"methods"
,"patient samples" :"methods"
,"patient selection" :"methods"
,"patient summary" :"results"
,"patient(s)" :"methods"
,"patient(s) and animal(s)" :"methods"
,"patient(s) and intervention(s)" :"methods"
,"patient, intervention, and results" :"results"
,"patient, methods and results" :"methods"
,"patient/method" :"methods"
,"patient/methods" :"methods"
,"patient/participants" :"methods"
,"patients" :"methods"
,"patients & method" :"methods"
,"patients & methods" :"methods"
,"patients & setting" :"methods"
,"patients (or participants)" :"methods"
,"patients and control subjects" :"methods"
,"patients and controls" :"methods"
,"patients and design" :"methods"
,"patients and intervention" :"methods"
,"patients and interventions" :"methods"
,"patients and main outcome measurements" :"methods"
,"patients and main outcome measures" :"methods"
,"patients and material" :"methods"
,"patients and materials" :"methods"
,"patients and measurements" :"methods"
,"patients and method" :"methods"
,"patients and methodology" :"methods"
,"patients and methods" :"methods"
,"patients and other participants" :"methods"
,"patients and others participants" :"methods"
,"patients and outcome measures" :"methods"
,"patients and participants" :"methods"
,"patients and results" :"results"
,"patients and setting" :"methods"
,"patients and settings" :"methods"
,"patients and study design" :"methods"
,"patients and subjects" :"methods"
,"patients and technique" :"methods"
,"patients and techniques" :"methods"
,"patients and treatment" :"methods"
,"patients or materials" :"methods"
,"patients or other participants" :"methods"
,"patients or others participants" :"methods"
,"patients or participants" :"methods"
,"patients participants" :"methods"
,"patients(s)" :"methods"
,"patients, design, and setting" :"methods"
,"patients, material and methods" :"methods"
,"patients, material, methods" :"methods"
,"patients, materials and methods" :"methods"
,"patients, materials, and methods" :"methods"
,"patients, method" :"methods"
,"patients, methods" :"methods"
,"patients, methods and results" :"results"
,"patients, methods, and results" :"results"
,"patients, participants" :"methods"
,"patients, subjects and methods" :"methods"
,"patients--methods" :"methods"
,"patients-methods" :"methods"
,"patients/design" :"methods"
,"patients/intervention" :"methods"
,"patients/interventions" :"methods"
,"patients/material and method" :"methods"
,"patients/material and methods" :"methods"
,"patients/materials and methods" :"methods"
,"patients/materials/methods" :"methods"
,"patients/method" :"methods"
,"patients/methods" :"methods"
,"patients/participants" :"methods"
,"patients/setting" :"methods"
,"patients/subjects" :"methods"
,"performance" :"methods"
,"period" :"methods"
,"period covered" :"methods"
,"period of study" :"methods"
,"personal experience" :"methods"
,"persons" :"methods"
,"perspective" :"conclusions"
,"perspective & time frame" :"methods"
,"perspective (values)" :"methods"
,"perspectives" :"conclusions"
,"perspectives and conclusion" :"conclusions"
,"perspectives and conclusions" :"conclusions"
,"perspectives and projects" :"conclusions"
,"pharmacokinetics" :"results"
,"phenomena of interest" :"methods"
,"phenomenon of interest" :"methods"
,"physical examination" :"methods"
,"physical properties" :"background"
,"place" :"methods"
,"place and duration" :"methods"
,"place and duration of study" :"methods"
,"place in therapy" :"conclusions"
,"plain language summary" :"conclusions"
,"points of consensus" :"conclusions"
,"policy implications" :"conclusions"
,"policy points" :"conclusions"
,"population" :"methods"
,"population and method" :"methods"
,"population and methods" :"methods"
,"population and sample" :"methods"
,"population and setting" :"methods"
,"population or sample" :"methods"
,"population studied" :"methods"
,"population, material and methods" :"methods"
,"population, sample, setting" :"methods"
,"population/sample" :"methods"
,"populations" :"methods"
,"populations and methods" :"methods"
,"positioning and anaesthesia" :"methods"
,"positions" :"conclusions"
,"possible complications" :"methods"
,"postoperative care" :"methods"
,"postoperative management" :"methods"
,"potential clinical relevance" :"conclusions"
,"potential difficulties" :"conclusions"
,"potential intervention" :"methods"
,"potential relevance" :"conclusions"
,"practical application" :"conclusions"
,"practical applications" :"conclusions"
,"practical attitude" :"conclusions"
,"practical implications" :"conclusions"
,"practical recommendations" :"conclusions"
,"practical relevance" :"background"
,"practice description" :"methods"
,"practice guideline" :"conclusions"
,"practice implication" :"conclusions"
,"practice implications" :"conclusions"
,"practice innovation" :"methods"
,"practice pattern examined" :"methods"
,"practitioner points" :"conclusions"
,"precis" :"conclusions"
,"predictor" :"methods"
,"predictor & outcome" :"methods"
,"predictor or factor" :"methods"
,"predictor variable" :"methods"
,"predictor variables" :"methods"
,"predictor, outcomes, & measurements" :"methods"
,"predictors" :"methods"
,"predictors & outcome" :"methods"
,"predictors & outcomes" :"methods"
,"preface" :"background"
,"preliminary results" :"results"
,"preliminary studies" :"background"
,"premise" :"objective"
,"premise of study" :"objective"
,"premise of the study" :"objective"
,"preoperative counseling and informed consent" :"methods"
,"preoperative work up" :"methods"
,"presentation" :"methods"
,"presentation of a case" :"methods"
,"presentation of case" :"methods"
,"presentation of cases" :"methods"
,"presentation of hypothesis" :"objective"
,"presentation of the case" :"methods"
,"presentation of the hypothesis" :"objective"
,"prevalence" :"background"
,"prevention" :"conclusions"
,"preventive measures" :"results"
,"primary and secondary outcome measures" :"methods"
,"primary and secondary outcomes" :"methods"
,"primary argument" :"background"
,"primary endpoint" :"methods"
,"primary findings" :"results"
,"primary funding source" :"background"
,"primary measurements" :"methods"
,"primary measures" :"methods"
,"primary objective" :"objective"
,"primary objectives" :"objective"
,"primary outcome" :"methods"
,"primary outcome measure" :"methods"
,"primary outcome measures" :"methods"
,"primary outcome variable" :"methods"
,"primary outcome variables" :"methods"
,"primary outcomes" :"methods"
,"primary practice setting" :"methods"
,"primary practice setting(s)" :"methods"
,"primary practice settings" :"methods"
,"primary purpose" :"objective"
,"primary results" :"results"
,"primary study objective" :"objective"
,"primary variables of interest" :"methods"
,"principal conclusion" :"conclusions"
,"principal conclusions" :"conclusions"
,"principal finding" :"results"
,"principal findings" :"results"
,"principal findings and conclusions" :"conclusions"
,"principal findings/conclusions" :"conclusions"
,"principal findings/methodology" :"results"
,"principal measurements" :"methods"
,"principal observations" :"results"
,"principal results" :"results"
,"principle" :"background"
,"principle conclusions" :"conclusions"
,"principle findings" :"results"
,"principle results" :"results"
,"principles" :"background"
,"principles and methods" :"methods"
,"probands" :"methods"
,"probands and methods" :"methods"
,"problem" :"objective"
,"problem addressed" :"objective"
,"problem and background" :"objective"
,"problem and method" :"objective"
,"problem and method of study" :"objective"
,"problem and objective" :"objective"
,"problem and purpose" :"objective"
,"problem assessed" :"objective"
,"problem being addressed" :"objective"
,"problem considered" :"objective"
,"problem identification" :"objective"
,"problem statement" :"objective"
,"problem statement and background" :"objective"
,"problem statement and purpose" :"objective"
,"problem/condition" :"objective"
,"problem/objective" :"objective"
,"problems" :"objective"
,"problems addressed" :"objective"
,"problems and aims" :"objective"
,"problems/objectives" :"objective"
,"procedure" :"methods"
,"procedure and results" :"results"
,"procedures" :"methods"
,"procedures and results" :"results"
,"process" :"methods"
,"prognosis" :"conclusions"
,"program" :"methods"
,"program description" :"methods"
,"program design" :"methods"
,"program evaluation" :"results"
,"programme description" :"methods"
,"programme evaluation" :"results"
,"progress" :"results"
,"project" :"methods"
,"project description" :"methods"
,"projects" :"methods"
,"prophylaxis" :"results"
,"proposal" :"methods"
,"proposals" :"methods"
,"proposed method" :"methods"
,"prospect" :"conclusions"
,"prospective study" :"methods"
,"prospects" :"conclusions"
,"prospects and projects" :"conclusions"
,"protocol" :"methods"
,"protocol registration" :"background"
,"public health action" :"conclusions"
,"public health actions" :"conclusions"
,"public health implications" :"conclusions"
,"purpose" :"objective"
,"purpose & methods" :"objective"
,"purpose and background" :"objective"
,"purpose and clinical relevance" :"conclusions"
,"purpose and design" :"objective"
,"purpose and experimental design" :"objective"
,"purpose and method" :"objective"
,"purpose and methods" :"objective"
,"purpose and objective" :"objective"
,"purpose and objectives" :"objective"
,"purpose and patients" :"objective"
,"purpose and patients and methods" :"objective"
,"purpose and setting" :"objective"
,"purpose of investigation" :"objective"
,"purpose of research" :"objective"
,"purpose of review" :"objective"
,"purpose of study" :"objective"
,"purpose of the investigation" :"objective"
,"purpose of the report" :"objective"
,"purpose of the research" :"objective"
,"purpose of the review" :"objective"
,"purpose of the study" :"objective"
,"purpose of the work" :"objective"
,"purpose of this review" :"objective"
,"purpose of this study" :"objective"
,"purpose, patients, and methods" :"objective"
,"purpose/aim" :"objective"
,"purpose/aim of the study" :"objective"
,"purpose/aims" :"objective"
,"purpose/background" :"objective"
,"purpose/hypothesis" :"objective"
,"purpose/method" :"objective"
,"purpose/methods" :"objective"
,"purpose/objective" :"objective"
,"purpose/objective(s)" :"objective"
,"purpose/objectives" :"objective"
,"purpose/question" :"objective"
,"purposes" :"objective"
,"purposes and clinical relevance" :"conclusions"
,"purposes of the study" :"objective"
,"purposes/objectives" :"objective"
,"qualifying statements" :"conclusions"
,"quality assessment" :"methods"
,"quality improvement plan" :"methods"
,"quality of evidence" :"methods"
,"quality problem" :"objective"
,"quality problem or issue" :"objective"
,"question" :"objective"
,"question addressed" :"objective"
,"question of the study" :"objective"
,"question under study" :"objective"
,"question/purposes" :"objective"
,"questioning" :"objective"
,"questions" :"objective"
,"questions under study" :"objective"
,"questions under study / principles" :"objective"
,"questions under study/principles" :"objective"
,"questions/hypotheses" :"objective"
,"questions/purpose" :"objective"
,"questions/purposes" :"objective"
,"randomisation" :"methods"
,"randomization" :"methods"
,"rational" :"background"
,"rational and objective" :"objective"
,"rational and objectives" :"objective"
,"rational, aims and objectives" :"objective"
,"rationale" :"background"
,"rationale & objective" :"objective"
,"rationale & objectives" :"objective"
,"rationale aims and objectives" :"objective"
,"rationale and aim" :"objective"
,"rationale and aims" :"objective"
,"rationale and aims of the study" :"objective"
,"rationale and background" :"background"
,"rationale and design" :"methods"
,"rationale and goals" :"objective"
,"rationale and hypothesis" :"objective"
,"rationale and method" :"methods"
,"rationale and methods" :"methods"
,"rationale and objective" :"objective"
,"rationale and objectives" :"objective"
,"rationale and purpose" :"objective"
,"rationale for study" :"objective"
,"rationale for the study" :"objective"
,"rationale for this study" :"objective"
,"rationale objective" :"objective"
,"rationale of the study" :"objective"
,"rationale, aim & objectives" :"objective"
,"rationale, aim and objective" :"objective"
,"rationale, aims and objective" :"objective"
,"rationale, aims and objectives" :"objective"
,"rationale/objectives" :"objective"
,"rationales and objectives" :"objective"
,"reason for performing study" :"background"
,"reason for performing the study" :"objective"
,"reason for study" :"background"
,"reasons for performing study" :"background"
,"reasons for performing study and objective" :"objective"
,"reasons for performing the study" :"background"
,"recent advances" :"background"
,"recent data" :"methods"
,"recent developments" :"background"
,"recent finding" :"results"
,"recent findings" :"results"
,"recent progress" :"results"
,"recent studies" :"methods"
,"recommendation" :"conclusions"
,"recommendation 1" :"conclusions"
,"recommendation 2" :"conclusions"
,"recommendation 3" :"conclusions"
,"recommendation and outlook" :"conclusions"
,"recommendation and perspective" :"conclusions"
,"recommendation and perspectives" :"conclusions"
,"recommendations" :"conclusions"
,"recommendations and conclusions" :"conclusions"
,"recommendations and outlook" :"conclusions"
,"recommendations and perspective" :"conclusions"
,"recommendations and perspectives" :"conclusions"
,"recommendations for clinical practice" :"conclusions"
,"reference test" :"methods"
,"reference test & measurements" :"methods"
,"reference test or outcome" :"methods"
,"reference tests" :"methods"
,"reference/citation" :"background"
,"reflections" :"conclusions"
,"registration" :"background"
,"registration details" :"background"
,"registration id in irct" :"background"
,"registration number" :"background"
,"rehabilitation" :"results"
,"relevance" :"conclusions"
,"relevance for clinical practice" :"conclusions"
,"relevance to clinical or professional practice" :"conclusions"
,"relevance to clinical practice" :"conclusions"
,"relevance to practice" :"conclusions"
,"relevance/impact" :"conclusions"
,"relevant changes" :"results"
,"relevant findings" :"results"
,"remarks" :"conclusions"
,"report" :"methods"
,"report of a case" :"methods"
,"report of cases" :"methods"
,"reporting period" :"methods"
,"reporting period covered" :"methods"
,"reports" :"methods"
,"requirements" :"methods"
,"research" :"methods"
,"research and design methods" :"methods"
,"research and methods" :"methods"
,"research approach" :"methods"
,"research design" :"methods"
,"research design & methods" :"methods"
,"research design and method" :"methods"
,"research design and methods" :"methods"
,"research design and methods and results" :"results"
,"research design and participants" :"methods"
,"research design and subjects" :"methods"
,"research design, subjects, and measures" :"methods"
,"research design, subjects, measures" :"methods"
,"research design/methods" :"methods"
,"research designs and methods" :"methods"
,"research findings" :"results"
,"research implications" :"conclusions"
,"research implications/limitations" :"conclusions"
,"research limitations" :"conclusions"
,"research limitations/implications" :"conclusions"
,"research method" :"methods"
,"research method and procedures" :"methods"
,"research method/design" :"methods"
,"research methodology" :"methods"
,"research methodology/design" :"methods"
,"research methods" :"methods"
,"research methods & procedures" :"methods"
,"research methods and procedure" :"methods"
,"research methods and procedures" :"methods"
,"research objective" :"objective"
,"research objectives" :"objective"
,"research problem" :"objective"
,"research question" :"objective"
,"research questions" :"objective"
,"research recommendations" :"conclusions"
,"research setting" :"methods"
,"research strategy" :"methods"
,"research topics" :"results"
,"research, design and methods" :"methods"
,"resolution" :"methods"
,"resolutions" :"methods"
,"resource" :"methods"
,"resources" :"methods"
,"respondents" :"methods"
,"response" :"results"
,"result" :"results"
,"result & conclusion" :"conclusions"
,"result & conclusions" :"conclusions"
,"result and conclusion" :"conclusions"
,"result and conclusions" :"conclusions"
,"result and discussion" :"conclusions"
,"result(s)" :"results"
,"result/conclusion" :"conclusions"
,"results" :"results"
,"results & conclusion" :"conclusions"
,"results & conclusions" :"conclusions"
,"results & discussion" :"conclusions"
,"results & interpretation" :"conclusions"
,"results and analysis" :"results"
,"results and comments" :"results"
,"results and comparison with existing methods" :"results"
,"results and complications" :"results"
,"results and conclusion" :"conclusions"
,"results and conclusions" :"conclusions"
,"results and discussion" :"conclusions"
,"results and discussions" :"conclusions"
,"results and findings" :"results"
,"results and implications" :"conclusions"
,"results and interpretation" :"conclusions"
,"results and interpretations" :"conclusions"
,"results and limitations" :"conclusions"
,"results and major conclusion" :"conclusions"
,"results and methods" :"results"
,"results and observations" :"results"
,"results and recommendations" :"conclusions"
,"results and significance" :"conclusions"
,"results and statistics" :"results"
,"results and ð¡onclusion" :"conclusions"
,"results and/or conclusions" :"conclusions"
,"results of base-case analysis" :"results"
,"results of data analysis" :"results"
,"results of data synthesis" :"results"
,"results of sensitivity analyses" :"results"
,"results of sensitivity analysis" :"results"
,"results of studies" :"results"
,"results of the study" :"results"
,"results(s)" :"results"
,"results, conclusion" :"conclusions"
,"results, discussion" :"results"
,"results-conclusions" :"conclusions"
,"results/conclusion" :"conclusions"
,"results/conclusions" :"conclusions"
,"results/discussion" :"conclusions"
,"results/findings" :"results"
,"results/interpretation" :"results"
,"results/outcome" :"results"
,"results/outcomes" :"results"
,"results/significance" :"results"
,"results/statistics" :"results"
,"results/summary" :"results"
,"review" :"methods"
,"review date" :"methods"
,"review findings" :"results"
,"review method" :"methods"
,"review methods" :"methods"
,"review of literature" :"methods"
,"review of the literature" :"methods"
,"review process" :"methods"
,"review registration" :"background"
,"review results" :"results"
,"review strategy" :"methods"
,"review summary" :"results"
,"reviewer" :"methods"
,"reviewer's conclusions" :"conclusions"
,"reviewers" :"methods"
,"reviewers' conclusions" :"conclusions"
,"risk factors" :"methods"
,"risk stratification" :"methods"
,"risk-adapted therapy" :"methods"
,"risks" :"methods"
,"safety" :"results"
,"sample" :"methods"
,"sample & setting" :"methods"
,"sample and method" :"methods"
,"sample and methodology" :"methods"
,"sample and methods" :"methods"
,"sample and setting" :"methods"
,"sample population" :"methods"
,"sample size" :"methods"
,"sample(s)" :"methods"
,"sample/setting" :"methods"
,"samples" :"methods"
,"samples and methods" :"methods"
,"sampling" :"methods"
,"sampling and method" :"methods"
,"sampling and methods" :"methods"
,"scientific background" :"background"
,"scientific question" :"objective"
,"scientific significance" :"conclusions"
,"scope" :"methods"
,"scope and background" :"background"
,"scope and conclusions" :"conclusions"
,"scope and purpose" :"objective"
,"scope of review" :"methods"
,"scope of the problem" :"background"
,"scope of the report" :"objective"
,"scope of the review" :"objective"
,"scope of the study" :"objective"
,"search method" :"methods"
,"search methods" :"methods"
,"search protocol" :"methods"
,"search strategies" :"methods"
,"search strategy" :"methods"
,"search strategy & selection criteria" :"methods"
,"search strategy & sources" :"methods"
,"search strategy and selection criteria" :"methods"
,"second case" :"methods"
,"secondary objective" :"objective"
,"secondary objectives" :"objective"
,"secondary outcome measure" :"methods"
,"secondary outcome measures" :"methods"
,"secondary outcomes" :"results"
,"selected highlights" :"conclusions"
,"selection" :"methods"
,"selection criteria" :"methods"
,"selection criteria for studies" :"methods"
,"selection of studies" :"methods"
,"selection procedure" :"methods"
,"selection procedures" :"methods"
,"series summary" :"methods"
,"setting" :"methods"
,"setting & design" :"methods"
,"setting & participants" :"methods"
,"setting & population" :"methods"
,"setting & sample" :"methods"
,"setting and design" :"methods"
,"setting and intervention" :"methods"
,"setting and interventions" :"methods"
,"setting and method" :"methods"
,"setting and methods" :"methods"
,"setting and objective" :"objective"
,"setting and objectives" :"objective"
,"setting and participants" :"methods"
,"setting and patient(s)" :"methods"
,"setting and patients" :"methods"
,"setting and population" :"methods"
,"setting and results" :"results"
,"setting and sample" :"methods"
,"setting and sample population" :"methods"
,"setting and study participants" :"methods"
,"setting and study population" :"methods"
,"setting and subject" :"methods"
,"setting and subjects" :"methods"
,"setting and type of participants" :"methods"
,"setting(s)" :"methods"
,"setting, design, and patients" :"methods"
,"setting, participants" :"methods"
,"setting, participants, and measurements" :"methods"
,"setting, patients" :"methods"
,"setting, population, & intervention" :"methods"
,"setting, subjects & interventions" :"methods"
,"setting/ participants" :"methods"
,"setting/design" :"methods"
,"setting/location" :"methods"
,"setting/participants" :"methods"
,"setting/participants/resources" :"methods"
,"setting/patients" :"methods"
,"setting/population" :"methods"
,"setting/sample" :"methods"
,"setting/subjects" :"methods"
,"settings" :"methods"
,"settings & design" :"methods"
,"settings & participants" :"methods"
,"settings and design" :"methods"
,"settings and design & materials and methods" :"methods"
,"settings and designs" :"methods"
,"settings and methods" :"methods"
,"settings and participants" :"methods"
,"settings and patients" :"methods"
,"settings and subjects" :"methods"
,"settings/location" :"methods"
,"settings/participants" :"methods"
,"settings/subjects" :"methods"
,"short introduction" :"objective"
,"short summary" :"objective"
,"side effects" :"results"
,"significance" :"conclusions"
,"significance and impact" :"conclusions"
,"significance and impact of study" :"conclusions"
,"significance and impact of the study" :"conclusions"
,"significance and impacts of the study" :"conclusions"
,"significance and importance of the study" :"conclusions"
,"significance and the impact of the study" :"conclusions"
,"significance of research" :"conclusions"
,"significance of results" :"conclusions"
,"significance of the research" :"conclusions"
,"significance of the study" :"conclusions"
,"significance/conclusion" :"conclusions"
,"significance/conclusions" :"conclusions"
,"significances" :"conclusions"
,"significant and impact of the study" :"conclusions"
,"significant outcomes" :"conclusions"
,"situation" :"background"
,"social implications" :"conclusions"
,"solution" :"methods"
,"solutions" :"methods"
,"source" :"methods"
,"source of data" :"methods"
,"source of funding" :"background"
,"source of information" :"methods"
,"sources" :"methods"
,"sources of data" :"methods"
,"sources of information" :"methods"
,"sources used" :"methods"
,"special features" :"methods"
,"specialty" :"methods"
,"specific aim" :"objective"
,"specific aims" :"objective"
,"specific objective" :"objective"
,"specific objectives" :"objective"
,"specific research challenge" :"objective"
,"specimens" :"methods"
,"speculation" :"conclusions"
,"sponsor" :"background"
,"sponsors" :"background"
,"sponsorship" :"background"
,"sponsorships" :"background"
,"standard radiological methods" :"methods"
,"standard treatment" :"methods"
,"standards" :"methods"
,"starting point" :"background"
,"startpoints" :"methods"
,"state of art" :"background"
,"state of art and perspectives" :"conclusions"
,"state of knowledge" :"background"
,"state of the art" :"background"
,"state of the art and perspectives" :"conclusions"
,"state of the problem" :"objective"
,"statement of conclusions" :"conclusions"
,"statement of problem" :"background"
,"statement of problem and rationale" :"background"
,"statement of problems" :"background"
,"statement of purpose" :"objective"
,"statement of the problem" :"background"
,"statements of the problem" :"background"
,"statistic" :"methods"
,"statistical" :"methods"
,"statistical analyses" :"methods"
,"statistical analyses performed" :"methods"
,"statistical analysis" :"methods"
,"statistical analysis and results" :"results"
,"statistical analysis performed" :"methods"
,"statistical analysis used" :"methods"
,"statistical method" :"methods"
,"statistical methods" :"methods"
,"statistical tests" :"methods"
,"statistics" :"methods"
,"statistics analysis" :"methods"
,"strategies" :"methods"
,"strategies for change" :"methods"
,"strategies for improvement" :"methods"
,"strategy" :"methods"
,"strategy for change" :"methods"
,"strength of recommendation" :"methods"
,"strength of recommendation grade" :"methods"
,"studies" :"methods"
,"studies included" :"methods"
,"studies reviewed" :"methods"
,"study" :"methods"
,"study aim" :"objective"
,"study aims" :"objective"
,"study and design" :"methods"
,"study answer" :"conclusions"
,"study appraisal" :"methods"
,"study appraisal and synthesis" :"methods"
,"study appraisal and synthesis methods" :"methods"
,"study area" :"methods"
,"study background" :"background"
,"study design" :"methods"
,"study design & methods" :"methods"
,"study design & setting" :"methods"
,"study design and data collection" :"methods"
,"study design and main outcome measures" :"methods"
,"study design and measurements" :"methods"
,"study design and method" :"methods"
,"study design and methodology" :"methods"
,"study design and methods" :"methods"
,"study design and objective" :"objective"
,"study design and objectives" :"objective"
,"study design and participants" :"methods"
,"study design and patients" :"methods"
,"study design and results" :"results"
,"study design and setting" :"methods"
,"study design and settings" :"methods"
,"study design and size" :"methods"
,"study design and subjects" :"methods"
,"study design materials and methods" :"methods"
,"study design, materials and methods" :"methods"
,"study design, patients, and methods" :"methods"
,"study design, setting & participants" :"methods"
,"study design, setting and subjects" :"methods"
,"study design, setting, & participants" :"methods"
,"study design, setting, and patients" :"methods"
,"study design, size and duration" :"methods"
,"study design, size duration" :"methods"
,"study design, size, and duration" :"methods"
,"study design, size, duration" :"methods"
,"study design/data collection" :"methods"
,"study design/data collection/extraction methods" :"methods"
,"study design/material and methods" :"methods"
,"study design/materials and methods" :"methods"
,"study design/method" :"methods"
,"study design/methods" :"methods"
,"study design/patients and methods" :"methods"
,"study design/results" :"results"
,"study design/setting" :"methods"
,"study designs" :"methods"
,"study designs and methods" :"methods"
,"study designs/materials and methods" :"methods"
,"study development & implementation" :"methods"
,"study eligibility" :"methods"
,"study eligibility criteria" :"methods"
,"study eligibility criteria, participants and interventions" :"methods"
,"study eligibility criteria, participants, and interventions" :"methods"
,"study factors" :"methods"
,"study findings" :"results"
,"study funding" :"background"
,"study funding/competing interest" :"background"
,"study funding/competing interest(s)" :"background"
,"study funding/competing interests" :"background"
,"study goal" :"objective"
,"study goals" :"objective"
,"study group" :"methods"
,"study group and methods" :"methods"
,"study groups" :"methods"
,"study hypothesis" :"objective"
,"study identification" :"methods"
,"study inclusion and exclusion criteria" :"methods"
,"study limitation" :"conclusions"
,"study limitations" :"conclusions"
,"study material" :"methods"
,"study method" :"methods"
,"study methods" :"methods"
,"study object" :"objective"
,"study objective" :"objective"
,"study objectives" :"objective"
,"study outcome" :"results"
,"study outcomes" :"results"
,"study participants" :"methods"
,"study participants and methods" :"methods"
,"study patients" :"methods"
,"study period" :"methods"
,"study perspective" :"methods"
,"study population" :"methods"
,"study population and design" :"methods"
,"study population and methods" :"methods"
,"study population and setting" :"methods"
,"study populations" :"methods"
,"study protocol" :"methods"
,"study purpose" :"objective"
,"study question" :"objective"
,"study questions" :"objective"
,"study rationale" :"objective"
,"study registration" :"background"
,"study results" :"results"
,"study sample" :"methods"
,"study samples" :"methods"
,"study selection" :"methods"
,"study selection & results" :"results"
,"study selection and data abstraction" :"methods"
,"study selection and data extraction" :"methods"
,"study selection and extraction" :"methods"
,"study selection criteria" :"methods"
,"study selection/data extraction" :"methods"
,"study selections" :"methods"
,"study setting" :"methods"
,"study setting/data sources" :"methods"
,"study site" :"methods"
,"study subjects" :"methods"
,"study subjects and methods" :"methods"
,"study type" :"methods"
,"study units" :"methods"
,"study variables" :"methods"
,"study-design" :"methods"
,"study/principles" :"objective"
,"subject" :"methods"
,"subject & methods" :"methods"
,"subject and method" :"methods"
,"subject and methods" :"methods"
,"subject objective" :"objective"
,"subject(s)" :"methods"
,"subject/methods" :"methods"
,"subjects" :"methods"
,"subjects & method" :"methods"
,"subjects & methods" :"methods"
,"subjects & setting" :"methods"
,"subjects (materials) and methods" :"methods"
,"subjects and design" :"methods"
,"subjects and intervention" :"methods"
,"subjects and interventions" :"methods"
,"subjects and main outcome measures" :"methods"
,"subjects and material" :"methods"
,"subjects and materials" :"methods"
,"subjects and measurements" :"methods"
,"subjects and measures" :"methods"
,"subjects and method" :"methods"
,"subjects and methodology" :"methods"
,"subjects and methods" :"methods"
,"subjects and outcome measures" :"methods"
,"subjects and participants" :"methods"
,"subjects and patients" :"methods"
,"subjects and results" :"results"
,"subjects and setting" :"methods"
,"subjects and settings" :"methods"
,"subjects and study design" :"methods"
,"subjects and treatment" :"methods"
,"subjects or participants" :"methods"
,"subjects, main outcome measures" :"methods"
,"subjects, material & methods" :"methods"
,"subjects, material and methods" :"methods"
,"subjects, materials and methods" :"methods"
,"subjects, participants" :"methods"
,"subjects/design" :"methods"
,"subjects/interventions" :"methods"
,"subjects/materials" :"methods"
,"subjects/method" :"methods"
,"subjects/methods" :"methods"
,"subjects/participants" :"methods"
,"subjects/patients" :"methods"
,"subjects/patients and methods" :"methods"
,"subjects/samples" :"methods"
,"subjects/setting" :"methods"
,"subjects/settings" :"methods"
,"suggestions" :"methods"
,"summary" :"conclusions"
,"summary and background" :"background"
,"summary and background data" :"background"
,"summary and conclusion" :"conclusions"
,"summary and conclusions" :"conclusions"
,"summary and discussion" :"conclusions"
,"summary answer" :"conclusions"
,"summary background" :"background"
,"summary background data" :"background"
,"summary introduction" :"objective"
,"summary objective" :"objective"
,"summary objectives" :"objective"
,"summary of background" :"background"
,"summary of background data" :"background"
,"summary of background data and objectives" :"objective"
,"summary of background information" :"background"
,"summary of case" :"methods"
,"summary of comment" :"results"
,"summary of data" :"methods"
,"summary of evidence" :"methods"
,"summary of findings" :"results"
,"summary of important findings" :"results"
,"summary of key points" :"conclusions"
,"summary of recommendations" :"conclusions"
,"summary of report" :"methods"
,"summary of results" :"results"
,"summary of review" :"results"
,"summary of the background data" :"background"
,"summary of the findings" :"results"
,"summary points" :"conclusions"
,"summary statement" :"conclusions"
,"summary statements" :"conclusions"
,"summary/conclusions" :"conclusions"
,"supplemental material" :"background"
,"supplementary information" :"background"
,"support" :"background"
,"surgery" :"methods"
,"surgical approach" :"methods"
,"surgical method" :"methods"
,"surgical procedure" :"methods"
,"surgical procedures" :"methods"
,"surgical technique" :"methods"
,"surgical treatment" :"methods"
,"survey" :"methods"
,"survey design & setting" :"methods"
,"survey instrument" :"methods"
,"survey sample" :"methods"
,"surveys" :"methods"
,"survival" :"results"
,"symptoms" :"methods"
,"synopsis" :"conclusions"
,"synthesis" :"results"
,"synthesis of evidence" :"results"
,"system description" :"methods"
,"systematic review methodology" :"methods"
,"systematic review registration" :"background"
,"tabulation, integration and results" :"results"
,"tabulation, integration, and results" :"results"
,"take home message" :"conclusions"
,"take home messages" :"conclusions"
,"take-home message" :"conclusions"
,"target" :"objective"
,"target audience" :"background"
,"target population" :"methods"
,"task force recommendations" :"conclusions"
,"taxonomy" :"methods"
,"teaching points" :"conclusions"
,"technical considerations" :"methods"
,"technical note" :"methods"
,"technique" :"methods"
,"techniques" :"methods"
,"technology" :"methods"
,"testing" :"methods"
,"testing of the hypothesis" :"methods"
,"testing the hypothesis" :"methods"
,"tests" :"methods"
,"the aim" :"objective"
,"the aim of our study" :"objective"
,"the aim of study" :"objective"
,"the aim of the paper" :"objective"
,"the aim of the study" :"objective"
,"the aim of the work" :"objective"
,"the aim of this study" :"objective"
,"the aim of this work" :"objective"
,"the aim of work" :"objective"
,"the aims" :"objective"
,"the aims of our study were" :"objective"
,"the case" :"methods"
,"the conclusion" :"conclusions"
,"the future" :"conclusions"
,"the goal" :"objective"
,"the hypothesis" :"objective"
,"the issue" :"objective"
,"the main aim of the study" :"objective"
,"the method" :"methods"
,"the object" :"objective"
,"the objective" :"objective"
,"the primary objective" :"objective"
,"the problem" :"objective"
,"the purpose" :"objective"
,"the purpose of the research" :"objective"
,"the purpose of the study" :"objective"
,"the purpose of this study" :"objective"
,"the research objective" :"objective"
,"the results" :"results"
,"the solution" :"methods"
,"the study objective" :"objective"
,"the technology" :"methods"
,"the technology being reviewed" :"methods"
,"theoretical background" :"background"
,"theoretical considerations" :"background"
,"theoretical framework" :"methods"
,"theories" :"background"
,"theory" :"background"
,"theory and methods" :"methods"
,"therapeutic implications" :"conclusions"
,"therapeutic management" :"conclusions"
,"therapeutic methods" :"methods"
,"therapy" :"methods"
,"therapy and clinical course" :"methods"
,"therapy and course" :"results"
,"therapy and outcome" :"results"
,"therapy and results" :"results"
,"this retrospective study aims" :"objective"
,"time horizon" :"methods"
,"timing" :"methods"
,"title" :"background"
,"tolerability" :"results"
,"tools" :"methods"
,"tools and methods" :"methods"
,"topic" :"objective"
,"topic of the study" :"objective"
,"toxicity" :"results"
,"toxicokinetics" :"methods"
,"transmission" :"methods"
,"treatment" :"methods"
,"treatment and clinical course" :"methods"
,"treatment and course" :"methods"
,"treatment and follow-up" :"results"
,"treatment and further course" :"methods"
,"treatment and methods" :"methods"
,"treatment and outcome" :"results"
,"treatment innovations" :"methods"
,"treatment protocol" :"methods"
,"treatment recommendations" :"conclusions"
,"treatment/outcome" :"results"
,"treatments" :"methods"
,"trial design" :"methods"
,"trial number" :"background"
,"trial register" :"background"
,"trial registration" :"background"
,"trial registration isrctn" :"background"
,"trial registration no" :"background"
,"trial registration number" :"background"
,"trial registration numbers" :"background"
,"trial registrations" :"background"
,"trial registry" :"background"
,"trial status" :"methods"
,"trials" :"methods"
,"trials registration" :"background"
,"tweetable abstract" :"conclusions"
,"type of participant" :"methods"
,"type of participants" :"methods"
,"type of review" :"methods"
,"type of studies reviewed" :"methods"
,"type of study" :"methods"
,"type of study/design" :"methods"
,"type of study/level of evidence" :"methods"
,"type of the study" :"methods"
,"types of participants" :"methods"
,"types of studies reviewed" :"methods"
,"unique information provided" :"conclusions"
,"uniqueness" :"results"
,"useful websites" :"background"
,"vaccination recommendations" :"conclusions"
,"validating the recommendations" :"methods"
,"validation" :"results"
,"validity" :"results"
,"validity and coverage" :"results"
,"value/originality" :"conclusions"
,"values" :"methods"
,"variables" :"methods"
,"variables measured" :"methods"
,"variables of interest" :"methods"
,"variables studied" :"methods"
,"veterinary data synthesis" :"results"
,"video abstract available" :"background"
,"viewpoint" :"conclusions"
,"viewpoint and conclusion" :"conclusions"
,"viewpoint and conclusions" :"conclusions"
,"viewpoints" :"results"
,"volunteers" :"methods"
,"volunteers and methods" :"methods"
,"what is already known" :"background"
,"what is already known about this subject" :"background"
,"what is known" :"background"
,"what is known already" :"background"
,"what is known and background" :"background"
,"what is known and objective" :"objective"
,"what is known and objectives" :"objective"
,"what is known and what this paper adds" :"background"
,"what is new and conclusion" :"conclusions"
,"what is new and conclusions" :"conclusions"
,"what the reader will gain" :"results"
,"what the readers will gain" :"results"
,"what this paper adds" :"conclusions"
,"what this study adds" :"conclusions"
,"what will the reader gain" :"background"
,"where next" :"conclusions"
,"wider implication of the findings" :"conclusions"
,"wider implications of the finding" :"conclusions"
,"wider implications of the findings" :"conclusions"
,"work method" :"methods"
,"work results" :"methods"
,"working hypothesis" :"objective"
,"setting" : "methods"
,"participants" : "methods" }

