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
import gc

INDEX_NAME = 'pubmed'

def doc_worker(input, output):
	for func,args in iter(input.get, 'STOP'):
		result = doc_calculate(func, args)
		output.put(result)

def doc_calculate(func, args):
	return func(*args)


def index_doc_from_elem(elem, filter_words_df, filename):

	if (is_issn(elem, '1533-4406') or is_issn(elem, '0028-4793') \
		or is_issn(elem, '0002-838X') or is_issn(elem, '1532-0650')\
		or is_issn(elem, '0003-4819') or is_issn(elem, '1539-3704')\
		or is_issn(elem, '0098-7484') or is_issn(elem, '1538-3598')):

		json_str = {}
		json_str = get_journal_info(elem, json_str)
		if json_str['journal_pub_year'] is not None:
			if (int(json_str['journal_pub_year']) > 1990):
	
				json_str = get_article_info(elem, json_str)
				
				if (not bool(set(json_str['article_type']) & set(['Letter', 'Editorial', 'Comment', 'Biography', 'Patient Education Handout', 'News']))):

					json_str = get_pmid(elem, json_str)

					json_str = get_article_ids(elem, json_str)
					
					json_str['citations_pmid'] = get_article_citations(elem)

					json_str['title_conceptids'] = get_snomed_annotation(json_str['article_title'], filter_words_df)
					json_str['abstract_conceptids'] = get_abstract_conceptids(json_str['article_abstract'], filter_words_df)
	
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
						es.index(index=INDEX_NAME, doc_type='abstract', body=json_obj)
						
						
					elif query_result['hits']['total'] == 1:
						article_id = query_result['hits']['hits'][0]['_id']
						es.index(index=INDEX_NAME, id=article_id, doc_type='abstract', body=json_obj)
					


def load_pubmed_updates_v2():
	print('called function')
	es = u.get_es_client()

	pool = Pool(processes=40)

	cursor = pg.return_postgres_cursor()
	
	filter_words_query = "select words from annotation.filter_words"
	filter_words_df = pg.return_df_from_query(cursor, filter_words_query, None, ["words"])
	cursor.close()

	index_exists = es.indices.exists(index=INDEX_NAME)
	if not index_exists:
		es.indices.create(index=INDEX_NAME, body={})


	s3 = boto3.resource('s3')
	bucket = s3.Bucket('pubmed-baseline-1')
	for object in bucket.objects.all():
		file_num = int(re.findall('medline17n(.*).xml', object.key)[0])

		if file_num >= 28:

			bucket.download_file(object.key, object.key)
			print(object.key)
		
			file_timer = u.Timer('file')

			tree = ET.parse(object.key)		
			root = tree.getroot()

			file_abstract_counter = 0

			for elem in root:
				if elem.tag == 'PubmedArticle':
					pool.apply_async(index_doc_from_elem, (elem, filter_words_df, object.key))
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
			os.remove(object.key)
			# tree = None
			# root = None
			gc.collect()		
			file_timer.stop()

	pool.close()
	pool.join()


def og_pubmed_test():

	cursor = pg.return_postgres_cursor()
	
	filter_words_query = "select words from annotation.filter_words"
	filter_words_df = pg.return_df_from_query(cursor, filter_words_query, None, ["words"])
	cursor.close()

	folder_arr = ['resources/production_baseline_2']


	for folder_path in folder_arr:
		file_counter = 0
		for filename in os.listdir(folder_path):
			file_timer = u.Timer('file')
			file_path = folder_path + '/' + filename

			tree = ET.parse(file_path)

			root = tree.getroot()

			file_abstract_counter = 0
			for elem in root:
				if elem.tag == 'PubmedArticle':

					# Filtering only for NEJM articles
					if (is_issn(elem, '1533-4406') or is_issn(elem, '0028-4793') \
						or is_issn(elem, '0002-838X') or is_issn(elem, '1532-0650')\
						or is_issn(elem, '0003-4819') or is_issn(elem, '1539-3704')\
						or is_issn(elem, '0098-7484') or is_issn(elem, '1538-3598')):

						json_str = {}
						json_str = get_journal_info(elem, json_str)
						
						if json_str['journal_pub_year'] is not None:
							if (int(json_str['journal_pub_year']) > 1990):
								d=u.Timer("article-timer")
								json_str = get_pmid(elem, json_str)
								json_str = get_article_ids(elem, json_str)
								
								json_str = get_article_info(elem, json_str)
								json_str['citations_pmid'] = get_article_citations(elem)
								json_str['title_conceptids'] = get_snomed_annotation(json_str['article_title'], filter_words_df)
								json_str['abstract_conceptids'] = get_abstract_conceptids(json_str['article_abstract'], filter_words_df)
								json_str['original_index_time'] = datetime.datetime.now()
								json_str['filename'] = filename
								pmid = json_str['pmid']
								json_str =json.dumps(json_str)
								json_obj = json.loads(json_str)
								d.stop()

								
				
			file_counter += 1
			if file_counter >= 1:
				break
			print(file_abstract_counter)
			file_timer.stop()



def load_pubmed_updates():
	es = u.get_es_client()
	cursor = pg.return_postgres_cursor()
	
	filter_words_query = "select words from annotation.filter_words"
	filter_words_df = pg.return_df_from_query(cursor, filter_words_query, None, ["words"])
	cursor.close()

	folder_arr = ['resources/production_baseline_1', 'resources/production_baseline_2', 'resources/production_updates_10_21_17']
	index_name = 'pubmed3'
	index_exists = es.indices.exists(index=index_name)
	for folder_path in folder_arr:

		for filename in os.listdir(folder_path):
			file_timer = u.Timer('file')
			file_path = folder_path + '/' + filename

			tree = ET.parse(file_path)

			root = tree.getroot()

			file_abstract_counter = 0
			for elem in root:
				if elem.tag == 'PubmedArticle':

					# Filtering only for NEJM articles
					if (is_issn(elem, '1533-4406') or is_issn(elem, '0028-4793') \
						or is_issn(elem, '0002-838X') or is_issn(elem, '1532-0650')\
						or is_issn(elem, '0003-4819') or is_issn(elem, '1539-3704')\
						or is_issn(elem, '0098-7484') or is_issn(elem, '1538-3598')):

						json_str = {}
						json_str = get_journal_info(elem, json_str)
						
						if json_str['journal_pub_year'] is not None:
							if (int(json_str['journal_pub_year']) > 1990):
								d=u.Timer("article-timer")
								json_str = get_pmid(elem, json_str)
								json_str = get_article_ids(elem, json_str)
								
								json_str = get_article_info(elem, json_str)
								json_str['citations_pmid'] = get_article_citations(elem)
								json_str['title_conceptids'] = get_snomed_annotation(json_str['article_title'], filter_words_df)
								json_str['abstract_conceptids'] = get_abstract_conceptids(json_str['article_abstract'], filter_words_df)
								json_str['original_index_time'] = datetime.datetime.now()
								json_str['filename'] = filename
								pmid = json_str['pmid']
								json_str =json.dumps(json_str)
								json_obj = json.loads(json_str)
								d.stop()

								if index_exists:
									get_article_query = {'_source': ['id', 'pmid'], 'query': {'constant_score': {'filter' : {'term' : {'pmid': pmid}}}}}
									query_result = es.search(index=index_name, body=get_article_query)

									if query_result['hits']['total'] == 0 or query_result['hits']['total'] > 1:
										es.index(index=index_name, doc_type='abstract', body=json_obj)
									
									elif query_result['hits']['total'] == 1:
										article_id = query_result['hits']['hits'][0]['_id']
										es.index(index=index_name, id=article_id, doc_type='abstract', body=json_obj)
									file_abstract_counter += 1
								else:
									es.index(index=index_name, doc_type='abstract', body=json_obj)
									index_exists = True
				elif elem.tag == 'DeleteCitation':

					delete_pmid_arr = get_deleted_pmid(elem)

					for pmid in delete_pmid_arr:
						get_article_query = {'_source': ['id', 'pmid'], 'query': {'constant_score': {'filter' : {'term' : {'pmid': pmid}}}}}
						query_result = es.search(index=index_name, body=get_article_query)

						if query_result['hits']['total'] == 0:
							continue
						elif query_result['hits']['total'] == 1:
							article_id = query_result['hits']['hits'][0]['_id']
							es.delete(index=index_name, doc_type='abstract', id=article_id)
						else:
							print("delete: more than one document found")
							print(pmid)
				
			print(file_abstract_counter)
			file_timer.stop()

def update_abstracts_with_conceptids():
	es =u.get_es_client()
	page = es.search(index='pubmed', doc_type='abstract', scroll='1000m', \
		size=1000, body={"query" : {"match_all" : {}}})

	counter = len(page['hits']['hits'])
	sid = page['_scroll_id']
	scroll_size = page['hits']['total']

	c=u.Timer(str(counter))
	abstract_conceptid_update_iterator(page)
	c.stop()

	while (scroll_size > 0):
		page = es.scroll(scroll_id = sid, scroll='1000m')
		sid = page['_scroll_id']
		scroll_size = len(page['hits']['hits'])

		counter += len(page['hits']['hits'])
		
		c = u.Timer(str(counter))
		abstract_conceptid_update_iterator(page)
		c.stop()


def abstract_conceptid_update_iterator(sr):
	cursor = pg.return_postgres_cursor()
	filter_words_query = "select words from annotation.filter_words"
	filter_words_df = pg.return_df_from_query(cursor, filter_words_query, None, ["words"])
	cursor.close()
	es = u.get_es_client()
	for abstract in sr['hits']['hits']:
		title_conceptids = get_abstract_title_conceptids(abstract['_source']['article_title'], \
			filter_words_df)
		abstract_conceptids = get_abstract_conceptids(abstract['_source']['article_abstract'], \
			filter_words_df)
		article_id = abstract['_id']

		conceptids = {}
		
		conceptids['doc'] = {"title_conceptids" : title_conceptids, \
			"abstract_conceptids" : abstract_conceptids}
		update_json_str = json.dumps(conceptids)
		update_json_obj = json.loads(update_json_str)
		es.update(index='pubmed', id=article_id, doc_type='abstract', body=update_json_obj)

def get_abstract_conceptids(abstract_dict, filter_words_df):
	result_dict = {}
	
	if abstract_dict is not None:
		for k1 in abstract_dict:
			sub_res_dict = {}
			for k2 in abstract_dict[k1]:
				sub_res_dict[k2] = get_snomed_annotation(abstract_dict[k1][k2], filter_words_df)
			result_dict[k1] = sub_res_dict

		return result_dict
	else:
		return None

def get_abstract_title_conceptids(abstract_title, filter_words_df):
	if abstract_title is not None:
		return get_snomed_annotation(abstract_title, filter_words_df)
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
			return annotation['conceptid'].tolist()
		else:
			return None

if __name__ == "__main__":
	t = u.Timer("full")
	# add_article_types()
	load_pubmed_updates_v2()
	t.stop()

	# tree = ET.parse('medline17n0028.xml')		
	# root = tree.getroot()
	# for elem in root:
	# 	if elem.tag == 'PubmedArticle':

	# 		if (is_issn(elem, '1533-4406') or is_issn(elem, '0028-4793') \
	# 			or is_issn(elem, '0002-838X') or is_issn(elem, '1532-0650')\
	# 			or is_issn(elem, '0003-4819') or is_issn(elem, '1539-3704')\
	# 			or is_issn(elem, '0098-7484') or is_issn(elem, '1538-3598')):
	# 			print('journal')
	# 			json_str = {}
	# 			json_str = get_journal_info(elem, json_str)
	# 			if json_str['journal_pub_year'] is not None:
	# 				if (int(json_str['journal_pub_year']) > 1990):
	# 					print('true')


