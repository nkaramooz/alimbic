from lxml import etree as ET
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
import copy
import re
import io
import sqlalchemy as sqla


def load_pubmed_local_2():
	start_file = 500
	folder_arr = ['resources/baseline']
	number_of_processes = 8
	
	task_queue = mp.Queue()

	pool = []
	
	for i in range(number_of_processes):
		p = mp.Process(target=doc_worker, args=(task_queue,))
		pool.append(p)
		p.start()

	for folder_path in folder_arr:
		file_counter = 0

		file_lst = os.listdir(folder_path)
		file_lst.sort()

		for filename in file_lst:
			print('start_file')
			abstract_counter = 0
			file_path = folder_path + '/' + filename

			file_num = int(re.findall('pubmed18n(.*).xml', filename)[0])

			if file_num >= start_file:
				print(filename)
			
				file_timer = u.Timer('file')

				# tree = ET.parse(file_path)		
				for event, element in ET.iterparse(file_path, tag="PubmedArticle"):
					json_str = {}
					params = (ET.tostring(element),json_str)
					task_queue.put((get_journal_info, params))
				
					element.clear()
					

				file_abstract_counter = 0

				file_timer.stop()
				root.clear()
				if file_num >= start_file+10:
					break
			print('end_file')	
		if file_num >= start_file+10:
			break

	for i in range(number_of_processes):
		task_queue.put('STOP')

	for p in pool:
		p.join()

def doc_worker(input):
	for func,args in iter(input.get, 'STOP'):
		doc_calculate(func, args)


def doc_calculate(func, args):
	func(*args)

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

def get_journal_info(elem, json_str):
	elem = ET.parse(io.BytesIO(elem))
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
	print(json_str)
	return json_str		


def write_jsonb():	
	conn,cursor = pg.return_postgres_cursor()
	engine = pg.return_sql_alchemy_engine()
	json_str = {"root" : "test", "child" : ["ace", "beta", "kappa"], "parents" : {"car1" : "null", "car2" : "bambie"}, "tup" : (1, '4')}

	json_str =json.dumps(json_str)
	json_obj = json.loads(json_str)
	cdf = pd.DataFrame([[json_obj]], columns=["field"])
	cdf['pd'] = 7778
	cdf.to_sql('tmp', engine, schema='annotation', if_exists='replace', index=False, dtype={'field' : sqla.types.JSON, 'pd' : sqla.types.String})

	# select field->> 'root' from annotation.tmp
	q = "select field from annotation.tmp"
	ts_df = pg.return_df_from_query(cursor, q, None, ['field'])
	u.pprint(type(ts_df['field'][0]['tup'][0]))

load_pubmed_local_2()

# write_jsonb()
