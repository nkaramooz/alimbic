import xml.etree.ElementTree as ET 
import json
from elasticsearch import Elasticsearch
from nltk.stem.wordnet import WordNetLemmatizer
import sys 
import codecs
import snomed_annotator as ann
import nltk.data
import pandas as pd
import multiprocessing as mp
import utils as u, pglib as pg
import os


def load_pubmed_updates():
	es = Elasticsearch([{'host' : 'localhost', 'port' : 9200}])
	
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
								json_str = get_pmid(elem, json_str)
								json_str = get_article_ids(elem, json_str)
								
								json_str = get_article_info(elem, json_str)
								json_str['citations_pmid'] = get_article_citations(elem)
								json_str['title_conceptids'] = get_snomed_annotation(json_str['article_title'], filter_words_df)
								json_str['abstract_conceptids'] = get_abstract_conceptids(json_str['article_abstract'], filter_words_df)
								
								pmid = json_str['pmid']
								json_str =json.dumps(json_str)
								json_obj = json.loads(json_str)

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
	es = Elasticsearch([{'host' : 'localhost', 'port' : 9200}])
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
	es = Elasticsearch([{'host' : 'localhost', 'port' : 9200}])

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
	
	if text is None:
		return None
	else:
		annotation = annotate_text(text, filter_words_df)

		if annotation is not None:
			return annotation['conceptid'].tolist()
		else:
			return None

def annotate_text(text, filter_words_df):
	number_of_processes = 8

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
	line = ann.clean_text(line)
	annotation = ann.return_line_snomed_annotation(cursor, line, 93, filter_words_df)
	cursor.close()
	return annotation

if __name__ == "__main__":
	t = u.Timer("full")
	# add_article_types()
	load_pubmed_updates()
	t.stop()

