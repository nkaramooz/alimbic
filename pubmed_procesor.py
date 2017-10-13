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

def load_pubmed():
	tree = ET.parse('resources/sample_data/medline17n0001.xml')

	root = tree.getroot()
	# es = Elasticsearch([{'host' : 'localhost', 'port' : 9200}])
	cursor = pg.return_postgres_cursor()
	filter_words_query = "select words from annotation.filter_words"
	filter_words_df = pg.return_df_from_query(cursor, filter_words_query, None, ["words"])

	for elem in root:
		# Filtering only for NEJM articles
		if is_issn(elem, '0028-4793'):
			t = u.Timer('article')
			json_str = {}
			json_str = get_pmid(elem, json_str)
			json_str = get_journal_info(elem, json_str)
			json_str = get_article_info(elem, json_str)
			json_str['title_conceptids'] = get_snomed_annotation(json_str['article_title'], filter_words_df)
			json_str['abstract_conceptids'] = get_snomed_annotation(json_str['article_abstract'], filter_words_df)
			json_str =json.dumps(json_str)
			json_obj = json.loads(json_str)
			t.stop()
			print(json_obj)
			
			# es.index(index='nlm', doc_type='pubmed', body=json_obj)

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

	try:
		title_elem = article_elem.findall('./ArticleTitle')[0]
		json_str['article_title'] = title_elem.text
	except:
		json_str['article_title'] = None

	try:
		abstract_elem = article_elem.findall('./Abstract/AbstractText')[0]
		json_str['article_abstract'] = abstract_elem.text
	except:
		json_str['article_abstract'] = None

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
	annotation = snomed.return_line_snomed_annotation(cursor, line, 100, filter_words_df)

	return annotation


if __name__ == "__main__":
	t = u.Timer("full")
	load_pubmed()
	t.stop()

