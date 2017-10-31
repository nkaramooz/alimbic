import pandas as pd
import re
import pickle
import sys
import psycopg2
from sqlalchemy import create_engine
from fuzzywuzzy import fuzz
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import time
import multiprocessing as mp
import copy
import utils as u, pglib as pg
import xml.etree.ElementTree as ET 
from elasticsearch import Elasticsearch
import os

def load_pubmed_baseline():
	folder_arr = ['resources/production_baseline_1', 'resources/production_baseline_1', 'resources/production_updates_10_21_17']
	
	for folder_path in folder_arr:

		for filename in os.listdir(folder_path):

			file_path = folder_path + '/' + filename

			tree = ET.parse(file_path)

			root = tree.getroot()

			file_abstract_counter = 0
			for elem in root:


				json_str = {}
				json_str = get_article_ids(elem, json_str)
				# json_str = get_pmid(elem, json_str)
				try:
					if json_str['article_ids']['pmid'] == '21864184':
						print(file_path)
						json_str = get_article_ids(elem, json_str)
						json_str = get_journal_info(elem, json_str)
						json_str = get_article_info(elem, json_str)
						json_str['citations_pmid'] = get_article_citations(elem)
						json_str =json.dumps(json_str)
						json_obj = json.loads(json_str)
						print(json_str)
						print(json_obj)
				except:
					print(elem.findall("."))

def get_pmid(elem, json_str):
	pmid = elem.findall('*/PMID')
	try:
		json_str['pmid'] = pmid[0].text
	except:
		json_str['pmid'] = None
	return json_str

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

if __name__ == "__main__":

	load_pubmed_baseline()