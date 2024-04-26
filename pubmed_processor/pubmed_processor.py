from lxml import etree as ET
import io
import json
from snomed_annotator import snomed_annotator as ann
from psycopg2 import pool
import pandas as pd
import multiprocessing as mp
import utilities.utils2 as u, utilities.pglib as pg
import sqlalchemy as sqla
import os
import datetime
from multiprocessing import Pool
import re
from nltk.stem.wordnet import WordNetLemmatizer
import regex as re
from time import sleep
import numpy as np
import os

INDEX_NAME = os.environ["INDEX_NAME"]
NUM_PROCESSES = int(os.environ["NUM_PROCESSES"])
CONCEPTS_OF_INTEREST = pg.get_all_concepts_of_interest()
CONDITIONS_OF_INTEREST = pg.get_all_conditions_set()
TREATMENTS_OF_INTEREST = pg.get_all_treatments_set()
DIAGNOSTICS_OF_INTEREST = pg.get_all_diagnostics_set()
CAUSES_OF_INTEREST = pg.get_all_causes_set()
STUDY_DESIGNS_OF_INTEREST = pg.get_all_study_designs_set()

month_dict = {
	"jan" : 1
	,"feb" : 2
	,"mar" : 3
	,"apr" : 4
	,"may" : 5
	,"jun" : 6
	,"jul" : 7
	,"aug" : 8
	,"sep" : 9
	,"oct" : 10
	,"nov" : 11
	,"dec" : 12
}


# This function is the main function that orchestrates indexing pubmed articles across
# each file. Leverages multiprocessing.
def load_pubmed(start_file, end_file, folder_path, two_char_year, lmtzr_list):
	es = u.get_es_client()
	task_queue = mp.Queue()
	pool = []

	for i in range(NUM_PROCESSES):
		p = mp.Process(target=doc_worker, args=(task_queue,lmtzr_list[i]))
		pool.append(p)
		p.start()

	# List of journal issns that will be indexed.
	issn_list = pg.return_df_from_query("select issn from pubmed.journals", None, ['issn'])['issn'].tolist()
	index_exists = es.indices.exists(index=INDEX_NAME)

	if not index_exists:
		# Initial schema for elasticsearch. 
		settings = {"mappings" :{"properties" : {
			"journal_issn" : {"type" : "keyword"}
			,"journal_issn_type" : {"type" : "keyword"}
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
			,"title_cids" : {"type" : "keyword"}
			,"title_dids" : {"type" : "keyword"}
			,"abstract_conceptids" : {"properties" : {"methods_cid" : {"type" : "keyword"}, 
				"background_cid" : {"type" : "keyword"},
				"conclusions_cid" : {"type" : "keyword"},
				"objective_cid" : {"type" : "keyword"},
				"results_cid" : {"type" : "keyword"},
				"unlabelled_cid" : {"type" : "keyword"},
				"keywords_cid" : {"type" : "keyword"}}}
			,"abstract_dids" : {"properties" : {"methods_did" : {"type" : "keyword"}, 
				"background_did" : {"type" : "keyword"},
				"conclusions_did" : {"type" : "keyword"},
				"objective_did" : {"type" : "keyword"},
				"results_did" : {"type" : "keyword"},
				"unlabelled_did" : {"type" : "keyword"},
				"keywords_did" : {"type" : "keyword"}}}
			,"concepts_of_interest" : {"type" : "keyword"}
			,"conditions_of_interest" : {"type" : "keyword"}
			,"treatments_of_interest" : {"type" : "keyword"}
			,"diagnostics_of_interest" : {"type" : "keyword"}
			,"causes_of_interest" : {"type" : "keyword"}
			,"study_designs_of_interest" : {"type" : "keyword"}
			,"article_type_id" : {"type" : "keyword"}
			,"article_type" : {"type" : "keyword"}
			,"index_date" : {"type" : "date", "format": "yyyy-MM-dd HH:mm:ss"}
			,"filename" : {"type" : "keyword"}
		}}}
		
		es.indices.create(index=INDEX_NAME, body=settings)
	file_counter = start_file

	while file_counter <= start_file+4 and file_counter <= end_file:
		print(file_counter)
		filename = 'pubmed%s' % two_char_year 
		filename += "n" + str(file_counter).zfill(4) + '.xml'
		file_path = folder_path + '/' + filename
		
		try:
			for event, elem in ET.iterparse(file_path, tag="PubmedArticle"):
				json_str = {}
				params = (ET.tostring(elem), filename, issn_list)
				task_queue.put((index_doc_from_elem, params))
				elem.clear()
			query = """
				INSERT INTO pubmed.indexed_files
				VALUES
				(%s, %s)
				"""
			pg.write_data(query, (filename, file_counter))
		except:
			print("File not found: " + file_path)
		file_counter += 1

	for i in range(NUM_PROCESSES):
		task_queue.put('STOP')

	for p in pool:
		p.join()


def doc_worker(input, lmtzr):
	for func,args in iter(input.get, 'STOP'):
		doc_calculate(func, args, lmtzr)


def doc_calculate(func, args, lmtzr):
	func(*args, lmtzr)


# Function to get abstract conceptids and write them to 
# the database and elasticsearch index. 
# TODO: Break this up into smaller functions.
def index_doc_from_elem(elem, filename, issn_list, lmtzr):
	elem = ET.parse(io.BytesIO(elem))
	issn = return_issn(elem)
	
	if issn in issn_list:
	
		json_str = {}
		json_str = get_journal_info(elem, json_str)

		if json_str['journal_pub_year'] is not None:

			if ((int(json_str['journal_pub_year']) >= 1985) and json_str['lang'] == 'eng'):
				json_str, article_text = get_article_info(elem, json_str)

				# Some abstracts in Pubmed don't have titles
				# Exclude protocols.
				if (json_str['article_title'] is not None and json_str['article_title'] != '') and ('a protocol for ' not in json_str['article_title'].lower() and \
					 ' protocol for an ' not in json_str['article_title'].lower() and \
					 ' protocol for a ' not in json_str['article_title'].lower() and \
					 'comment on' not in json_str['article_title'].lower() and \
					 'corrigendum' not in json_str['article_title'].lower()):
					if (not bool(set(json_str['article_type']) & \
						set(['Letter', 'Erratum to', 'Editorial', 'Comment', 'Commentary', 'Biography', 'Patient Education Handout', \
								'News', 'Published Erratum', 'Clinical Trial Protocol', 'Retraction of Publication',\
								'Retracted Publication', 'Clinical Trial Protocol', 'Research Design', 'Duplicate Publication', \
								'Expression of Concern', 'Interview', 'Legal Case', 'Newspaper Article', 'Personal Narrative', \
								'Portrait', 'Video-Audio Media', 'Webcast']))):

						json_str = get_pmid(elem, json_str)
						json_str = get_article_ids(elem, json_str)					
						json_str['citations_pmid'] = get_article_citations(elem)
						pmid = json_str['pmid']
							
						annotation_dict, sentence_annotations_df, sentence_tuples_df, sentence_concept_arr_df = \
							get_abstract_conceptids(json_str, article_text, lmtzr)

						sentence_annotations_df['ver'] = 0
						sentence_concept_arr_df['ver'] = 0
						sentence_tuples_df['ver'] = 0

						# Store expected pivot categories in a separate section of the document
	  					# which will improve the speed when performing aggregations over these categories
						# during search.
						json_str['concepts_of_interest'] = []
						json_str['conditions_of_interest'] = []
						json_str['treatments_of_interest'] = []
						json_str['diagnostics_of_interest'] = []
						json_str['causes_of_interest'] = []
						json_str['study_designs_of_interest'] = []
						
						if annotation_dict['abstract'] is not None:
							json_str['abstract_conceptids'] = annotation_dict['abstract']['cid_dict']
							json_str['abstract_dids'] = annotation_dict['abstract']['did_dict']
							concept_list = annotation_dict['abstract']['cid_dict'].values()
							concept_list = [i for sublist in concept_list for i in sublist if i != -1]
							json_str['concepts_of_interest'].extend(list(CONCEPTS_OF_INTEREST & set(concept_list)))
							json_str['conditions_of_interest'].extend(list(CONDITIONS_OF_INTEREST & set(concept_list)))
							json_str['treatments_of_interest'].extend(list(TREATMENTS_OF_INTEREST & set(concept_list)))
							json_str['diagnostics_of_interest'].extend(list(DIAGNOSTICS_OF_INTEREST & set(concept_list)))
							json_str['causes_of_interest'].extend(list(CAUSES_OF_INTEREST & set(concept_list)))
							json_str['study_designs_of_interest'].extend(list(STUDY_DESIGNS_OF_INTEREST & set(concept_list)))

						else:
							json_str['abstract_conceptids'] = None
							json_str['abstract_dids'] = None
							
						if annotation_dict['title'] is not None:
							title_cids = annotation_dict['title']['cids']
							title_dids = annotation_dict['title']['dids']
						else:
							title_cids = None
							title_dids = None

						if title_cids is not None:
							json_str['title_cids'] = title_cids
							json_str['title_dids'] = title_dids
							json_str['concepts_of_interest'].extend(list(CONCEPTS_OF_INTEREST & set(title_cids)))
							json_str['conditions_of_interest'].extend(list(CONDITIONS_OF_INTEREST & set(title_cids)))
							json_str['treatments_of_interest'].extend(list(TREATMENTS_OF_INTEREST & set(title_cids)))
							json_str['diagnostics_of_interest'].extend(list(DIAGNOSTICS_OF_INTEREST & set(title_cids)))
							json_str['causes_of_interest'].extend(list(CAUSES_OF_INTEREST & set(title_cids)))
							json_str['study_designs_of_interest'].extend(list(STUDY_DESIGNS_OF_INTEREST & set(title_cids)))
						else:
							json_str['title_cids'] = None
							json_str['title_dids'] = None


						if annotation_dict['article_keywords'] is not None:
							keywords_cids = annotation_dict['article_keywords']['cids']
							keywords_dids = annotation_dict['article_keywords']['dids']
						else:
							keywords_cids = None
							keywords_dids = None


						if keywords_cids is not None:
							json_str['keywords_cids'] = keywords_cids
							json_str['keywords_dids'] = keywords_dids
							json_str['concepts_of_interest'].extend(list(CONCEPTS_OF_INTEREST & set(keywords_cids)))
							json_str['conditions_of_interest'].extend(list(CONDITIONS_OF_INTEREST & set(keywords_cids)))
							json_str['treatments_of_interest'].extend(list(TREATMENTS_OF_INTEREST & set(keywords_cids)))
							json_str['diagnostics_of_interest'].extend(list(DIAGNOSTICS_OF_INTEREST & set(keywords_cids)))
							json_str['causes_of_interest'].extend(list(CAUSES_OF_INTEREST & set(keywords_cids)))
							json_str['study_designs_of_interest'].extend(list(STUDY_DESIGNS_OF_INTEREST & set(keywords_cids)))
						else:
							json_str['keywords_cids'] = None
							json_str['keywords_dids'] = None

						json_str['index_date'] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
						json_str['filename'] = filename
												
						sentence_concept_arr = sentence_concept_arr_df['concept_arr'].tolist()
						abstract_concept_arr = [i for sublist in sentence_concept_arr for i in sublist]
						abstract_concept_df = pd.DataFrame([[json_str['pmid'], abstract_concept_arr, json_str['journal_pub_month'], \
							json_str['journal_pub_year'], json_str['journal_iso_abbrev']]], \
							columns=['pmid', 'abs_concept_arr', 'journal_pub_month', 'journal_pub_year',  'journal_iso_abbrev'])
						
						abstract_sentence_tuples = sentence_tuples_df['sentence_tuples'].tolist()
						abstract_sentence_tuples_arr = [i for sublist in abstract_sentence_tuples for i in sublist]
						abstract_og_sentence_tuples = sentence_tuples_df['og_sentence_tuples'].tolist()
						abstract_og_sentence_tuples_arr = [i for sublist in abstract_og_sentence_tuples for i in sublist]
						abstract_sentence_df = pd.DataFrame([[json_str['pmid'], abstract_sentence_tuples_arr, abstract_og_sentence_tuples_arr,\
							json_str['journal_pub_month'], json_str['journal_pub_year'], json_str['journal_iso_abbrev']]], \
							columns=['pmid', 'abstract_tuples', 'abstract_og_tuples', 'journal_pub_month', 'journal_pub_year', 'journal_iso_abbrev'])

						json_str =json.dumps(json_str)
						json_obj = json.loads(json_str)
						
						es = u.get_es_client()
						get_article_query = {'_source': ['id', 'pmid'], 'query': {'constant_score': {'filter' : {'term' : {'pmid': pmid}}}}}

						query_result = es.search(index=INDEX_NAME, body=get_article_query)
						
						# Verifies that the pmid does not already exist in the index.
						if query_result['hits']['total']['value'] == 0:
							try:
								es.index(index=INDEX_NAME, body=json_obj)
							except:
								print(json_obj)
								raise ValueError('incompatible json obj')
							engine = pg.return_sql_alchemy_engine()

							# Multiple try blocks included due to processing errors that resulted when using
							# 48 threads which would result in the database locking. Adding these additional try blocks
	   						# solved the problem temporarily, but would address this differently for a production tool.
							counter = 0
							while counter < 3:
								try:
									sentence_tuples_df.to_sql('sentence_tuples', engine, schema='pubmed', \
										if_exists='append', index=False, dtype={'sentence_tuples' : sqla.types.JSON, 'og_sentence_tuples' : sqla.types.JSON})
									sentence_annotations_df.to_sql('sentence_annotations', engine, schema='pubmed', \
										if_exists='append', index=False)
									sentence_concept_arr_df.to_sql('sentence_concept_arr', engine, schema='pubmed', \
										if_exists='append', index=False)
									abstract_concept_df.to_sql('abstract_concept_arr', engine, schema='pubmed', if_exists='append', index=False)
									abstract_sentence_df.to_sql('abstract_tuples', engine, schema='pubmed', \
										if_exists='append', index=False, dtype={'abstract_tuples' : sqla.types.JSON, 'abstract_og_tuples' : sqla.types.JSON})
									engine.dispose()
									break
								except:
									counter += 1
									sleep(0.5)
									engine.dispose()
				
						elif query_result['hits']['total']['value'] >= 1:
	   						#TODO Update the tables with new values for a new document
							article_id = query_result['hits']['hits'][0]['_id']
							es.index(index=INDEX_NAME, id=article_id, body=json_obj)
							
				
# Function called for each abstract to get the conceptids. 
def get_abstract_conceptids(abstract_dict, article_text, lmtzr):
	cid_dict = {}
	did_dict = {}
	result_dict = {}

	cleaned_text = ann.clean_text(article_text)

	all_words = ann.get_all_words_list(cleaned_text)

	cache = ann.get_cache(all_words_list=all_words, case_sensitive=True, \
			check_pos=False, spellcheck_threshold=100, lmtzr=lmtzr)

	sentence_annotations_df, sentence_tuples_df, sentence_concept_arr_df = \
		get_snomed_annotation(abstract_dict, cache, lmtzr)
	
	# Tech debt: The refactoring of the snomed_annotator script to use a_cid and a_did
	# for fuzzy matching purposes, has numerous downstream effects to eventually address.
	# For now, to preserve functionality will rename the output column back to acid and adid.
	# This doesn't have any end user impact. We would just need to update all the sql tables eventually
	# to use a_cid and a_did.
	sentence_annotations_df.rename(columns={'a_cid' : 'acid'}, inplace=True)
	sentence_annotations_df.rename(columns={'a_did' : 'adid'}, inplace=True)

	sentence_annotations_df['pmid'] = abstract_dict['pmid']
	sentence_tuples_df['pmid'] = abstract_dict['pmid']
	sentence_concept_arr_df['pmid'] = abstract_dict['pmid']

	sentence_annotations_df['journal_pub_month'] = abstract_dict['journal_pub_month']
	sentence_tuples_df['journal_pub_month'] = abstract_dict['journal_pub_month']
	sentence_concept_arr_df['journal_pub_month'] = abstract_dict['journal_pub_month']
	sentence_annotations_df['journal_pub_year'] = abstract_dict['journal_pub_year']
	sentence_tuples_df['journal_pub_year'] = abstract_dict['journal_pub_year']
	sentence_concept_arr_df['journal_pub_year'] = abstract_dict['journal_pub_year']
	
	sentence_annotations_df['journal_iso_abbrev'] = abstract_dict['journal_iso_abbrev']
	sentence_tuples_df['journal_iso_abbrev'] = abstract_dict['journal_iso_abbrev']
	sentence_concept_arr_df['journal_iso_abbrev'] = abstract_dict['journal_iso_abbrev']

	title_cids = sentence_annotations_df[(sentence_annotations_df['section']== 'article_title') 
		& (sentence_annotations_df['acid'] != '-1')]['acid'].tolist()
	title_dids = sentence_annotations_df[(sentence_annotations_df['section']== 'article_title') 
		& (sentence_annotations_df['adid'] != '-1')]['adid'].tolist()

	keywords_cids = sentence_annotations_df[(sentence_annotations_df['section']== 'article_keywords') 
		& (sentence_annotations_df['acid'] != '-1')]['acid'].tolist()
	keywords_dids = sentence_annotations_df[(sentence_annotations_df['section']== 'article_keywords') 
		& (sentence_annotations_df['adid'] != '-1')]['adid'].tolist()

	result_dict['title'] = {'cids' : title_cids, 'dids' : title_dids}
	result_dict['article_keywords'] = {'cids' : keywords_cids, 'dids' : keywords_dids}
	
	positive = sentence_annotations_df[(sentence_annotations_df['adid']=='-1')]['adid'].tolist()
	negative = sentence_annotations_df[~(sentence_annotations_df['adid']=='-1')]['adid'].tolist()
	
	if abstract_dict['article_abstract'] is not None:
		for index,k1 in enumerate(abstract_dict['article_abstract']):
			k1_cid = str(k1) + "_cid"
			k1_did = str(k1) + "_did"
			cid_dict[k1_cid] = sentence_annotations_df[(sentence_annotations_df['section']== k1) & (sentence_annotations_df['acid'] != '-1')]['acid'].tolist()
			did_dict[k1_did] = sentence_annotations_df[(sentence_annotations_df['section']== k1) & (sentence_annotations_df['adid'] != '-1')]['adid'].tolist()
		result_dict['abstract'] = {'cid_dict' : cid_dict, 'did_dict' : did_dict}
		return result_dict, sentence_annotations_df, sentence_tuples_df, sentence_concept_arr_df
	else:
		result_dict['abstract'] = None
		return result_dict, sentence_annotations_df, sentence_tuples_df, sentence_concept_arr_df


def get_snomed_annotation(text_dict, cache, lmtzr):
	sentences_df = pd.DataFrame()
	sentences_df = ann.return_section_sentences(text_dict['article_title'], 'article_title', 0, sentences_df)

	if text_dict['article_abstract'] is not None:
		for ind,k1 in enumerate(text_dict['article_abstract']):
			sentences_df = ann.return_section_sentences(text_dict['article_abstract'][k1], str(k1), ind+1, sentences_df)

	if text_dict['article_keywords'] is not None:
		sentences_df = ann.return_section_sentences(text_dict['article_keywords'], 'article_keywords', 0, sentences_df)

	sentences_df['line'] = sentences_df['line'].apply(lambda x: ann.clean_text(x))

	sentence_annotations_df, sentence_tuples_df, sentence_concept_arr_df = \
			ann.annotate_text(sentences_df=sentences_df, cache=cache, \
				case_sensitive=True, check_pos=False, acr_check=True, \
				return_details=True, lmtzr=lmtzr, spellcheck_threshold=100)
	return sentence_annotations_df, sentence_tuples_df, sentence_concept_arr_df


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
		return True if issn_text == issn else False
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
		issn_elem = journal_elem.findall('./ISSN')[0]
		json_str['journal_issn'] = issn_elem.text
		json_str['journal_issn_type'] = issn_elem.attrib['IssnType']
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
		month_text = str(month_elem.text).lower()
		if month_text in month_dict.keys():
			json_str['journal_pub_month'] = month_dict[month_text]
		else:
			json_str['journal_pub_month'] = int(month_text)
	except:
		json_str['journal_pub_month'] = None

	try: 
		day_elem = journal_elem.findall('./JournalIssue/PubDate/Day')[0]
		json_str['journal_pub_day'] = day_elem.text
	except:
		json_str['journal_pub_day'] = None

	try:
		lang = elem.findall('./MedlineCitation/Article/Language')[0]
		lang = lang.text 
		json_str['lang'] = lang
	except:
		json_str['lang'] = None

	return json_str


def get_elem_text(elem):
	return ''.join(elem.itertext())


# Update the json_str to include the structured data elements of interest
# from the xml element.
def get_article_info(elem, json_str):
	article_text = ""
	try:
		article_elem = elem.find('./MedlineCitation/Article')
	except:
		json_str['article_title'] = None
		json_str['article_abstract'] = None
		json_str['article_type'] = None
		json_str['article_type_id'] = None
		json_str['article_keywords'] = None

	try:
		title_elem = article_elem.find('./ArticleTitle')
		cleaned_title = get_elem_text(title_elem)
		json_str['article_title'] = cleaned_title
		article_text += cleaned_title
	except:
		json_str['article_title'] = None

	try:
		keyword_elem = elem.find('./MedlineCitation/KeywordList')

		for keyword in keyword_elem:
			cleaned_keywords = get_elem_text(keyword)
			if json_str['article_keywords'] == None:
				json_str['article_keywords'] = cleaned_keywords
			else:
				json_str['article_keywords'] = json_str['article_keywords'] + ' ' + cleaned_keywords
		article_text += cleaned_keywords
	except:
		json_str['article_keywords'] = None

	if article_elem.find('./Abstract') is not None:
		abstract_elem = article_elem.find('./Abstract')
		abstract_dict = {}

		for abstract_sub_elem in abstract_elem:
			if not abstract_sub_elem.attrib:
				cleaned_unlabelled = get_elem_text(abstract_sub_elem)
				if 'unlabelled' not in abstract_dict.keys():
					abstract_dict['unlabelled'] = cleaned_unlabelled
				else:
					abstract_dict['unlabelled'] = abstract_dict['unlabelled'] + "\r" + cleaned_unlabelled
				article_text += ' ' + cleaned_unlabelled
			else:			
				cleaned_abstract_sub_elem = get_elem_text(abstract_sub_elem)
				nlm_cat_dict = get_nlm_cat_dict()
				if abstract_sub_elem.attrib['Label'].lower() in nlm_cat_dict.keys():
					if nlm_cat_dict[abstract_sub_elem.attrib['Label'].lower()] in abstract_dict.keys():
						abstract_dict[nlm_cat_dict[abstract_sub_elem.attrib['Label'].lower()]] = \
							abstract_dict[nlm_cat_dict[abstract_sub_elem.attrib['Label'].lower()]] + "\r" + cleaned_abstract_sub_elem
					else:
						abstract_dict[nlm_cat_dict[abstract_sub_elem.attrib['Label'].lower()]] = cleaned_abstract_sub_elem
				elif 'NlmCategory' in abstract_sub_elem.attrib.keys() and abstract_sub_elem.attrib['NlmCategory'] != 'UNASSIGNED':
					if abstract_sub_elem.attrib['NlmCategory'].lower() in abstract_dict.keys():
						abstract_dict[abstract_sub_elem.attrib['NlmCategory'].lower()] = \
							abstract_dict[abstract_sub_elem.attrib['NlmCategory'].lower()] + "\r" + cleaned_abstract_sub_elem
					else:
						abstract_dict[abstract_sub_elem.attrib['NlmCategory'].lower()] = cleaned_abstract_sub_elem
				else:
					if 'unlabelled' not in abstract_dict.keys():
						abstract_dict['unlabelled'] = cleaned_abstract_sub_elem
					else:
						abstract_dict['unlabelled'] = abstract_dict['unlabelled'] + "\r" + cleaned_abstract_sub_elem
				article_text += ' ' + cleaned_abstract_sub_elem

		json_str['article_abstract'] = abstract_dict

	else:
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


# Function can be used to determine the starting file number to index.
def get_start_file_num():
	query = """
		select max(file_num) from pubmed.indexed_files
	"""
	max_num = pg.return_df_from_query(query, None, ['max'])['max']
	return 0 if max_num == None else max_num+1	


def return_lemmatizers_list():
	lmtzr_list = []
	for i in range(NUM_PROCESSES):
		lmtzr = WordNetLemmatizer()
		lmtzr_list.append(lmtzr)
	return lmtzr_list


# Get the mapping json that maps NLM sections
# to a smaller standardized set of sections.
def get_nlm_cat_dict():
	with open ("./pubmed_processor/nlm_categories.json", 'r') as cats:
		return json.load(cats)


if __name__ == "__main__":
	print("This script will index pubmed articles from the provided folder path.")
	# resource_path = str(input("Enter the relative path to the pubmed files (ex: resources/pubmed_baseline/ftp.ncbi.nlm.nih.gov/baseline): "))

	# two_char_year = str(input("Enter the two character year that will be used to identify the pubmed files to process\
	# 				   Example file: pubmed24n0690.xml) : "))
	# start_file = int(input("Enter the starting file number to index: "))
	# end_file = int(input("Enter the ending file number to index: "))
	
	lmtzr_list = return_lemmatizers_list()
	start_file = 1
	end_file = 1
	resource_path = "./pubmed_processor/test_file"
	two_char_year = "24"

	while (start_file <= end_file):
		load_pubmed(start_file, end_file, resource_path, two_char_year, lmtzr_list)
		start_file += 5
 



