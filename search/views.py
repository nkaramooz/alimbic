from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
import snomed_annotator as ann
import pglib as pg
from elasticsearch import Elasticsearch
from nltk.stem.wordnet import WordNetLemmatizer
import utils as u
import pandas as pd
# Create your views here.



def home_page(request):
	return render(request, 'search/home_page.html')

### ELASTIC SEARCH


def elastic_search_home_page(request):
	return render(request, 'search/elastic_home_page.html')


def post_elastic_search(request):
	query = request.POST['query']
	return HttpResponseRedirect(reverse('search:elastic_search_results', args=(query,)))

def elastic_search_results(request, query):
	es = Elasticsearch([{'host' : 'localhost', 'port' : 9200}])

	es_query = get_text_query(query)
	sr = es.search(index='pubmed', body=es_query)['hits']['hits']
	return render(request, 'search/elastic_search_results_page.html', {'sr' : sr, 'query' : query})


### CONCEPT SEARCH

def concept_search_home_page(request):
	return render(request, 'search/concept_search_home_page.html')

def post_concept_search(request):
	query = request.POST['query']
	return HttpResponseRedirect(reverse('search:concept_search_results', args=(query,)))

def concept_search_results(request, query):
	es = Elasticsearch([{'host' : 'localhost', 'port' : 9200}])
	cursor = pg.return_postgres_cursor()
	filter_words_query = "select words from annotation.filter_words"
	filter_words_df = pg.return_df_from_query(cursor, filter_words_query, None, ["words"])
	query_concepts_df = ann.return_line_snomed_annotation(cursor, query, 90, filter_words_df)

	sr = {}
	query_concepts_dict = None
	if query_concepts_df is not None:
		unmatched_terms = get_unmatched_terms(query, query_concepts_df, filter_words_df)
		full_query_concepts_list = ann.get_concept_synonyms_from_series(query_concepts_df['conceptid'], cursor)
		query_concepts_dict = get_query_arr_dict(full_query_concepts_list)

		es_query = {"from" : 0, \
				 "size" : 20, \
				 "query": get_query(full_query_concepts_list, unmatched_terms, cursor)}
		print(es_query)
		sr = es.search(index='pubmed', body=es_query)
		print(sr['hits']['hits'])
	else:
		es_query = get_text_query(query)
		sr = es.search(index='pubmed', body=es_query)
		print("TEXT QUERY")
	sr = sr['hits']['hits']

	if len(sr) == 0:
		print("TEXT QUERY")
		es_query = get_text_query(query)
		sr = es.search(index='pubmed', body=es_query)
		sr = sr['hits']['hits']

	# title_df = pd.DataFrame(sr['title_conceptids'], columns=['conceptid'])
	sr = add_concept_names_to_sr(sr)

	return render(request, 'search/concept_search_results_page.html', {'sr' : sr, 'query' : query, 'concepts' : query_concepts_dict})




### Utility functions

def get_unmatched_terms(query, query_concepts_df, filter_words_df):
	unmatched_terms = ""
	for index,word in enumerate(query.split()):
		if (filter_words_df['words'] == word).any():
			continue
		elif len(query_concepts_df[(query_concepts_df['term_end_index'] >= index) & (query_concepts_df['term_start_index'] <= index)]) > 0:
			continue
		else:
			unmatched_terms += word + " "
	return unmatched_terms

def add_concept_names_to_sr(sr):
	for hit in sr:
		try:
			title_df = pd.DataFrame(hit['_source']['title_conceptids'], columns=['conceptid'])
			title_df = ann.add_names(title_df)
			hit['_source']['title_conceptids'] = get_abstract_concept_arr_dict(title_df)
		except:
			hit['_source']['title_conceptids'] = None

		try:
			hit['_source']['abstract_conceptids'] = get_flattened_abstract_concepts(hit)
		except:
			hit['_source']['title_conceptids'] = None
	
	return sr

def get_flattened_abstract_concepts(hit):
	abstract_concept_arr = []
	try:
		for key1 in hit['_source']['abstract_conceptids']:
			for key2 in hit['_source']['abstract_conceptids'][key1]:
				concept_df = pd.DataFrame(hit['_source']['abstract_conceptids'][key1][key2], columns=['conceptid'])
				concept_df = ann.add_names(concept_df)
				concept_dict_arr = get_abstract_concept_arr_dict(concept_df)
				abstract_concept_arr.extend(concept_dict_arr)
	except:
		return None
	if len(abstract_concept_arr) == 0:
		return None
	else:
		return abstract_concept_arr

def get_article_type_filters():
	filt = [ \
			{"match": {"article_type" : "Letter"}}, \
			{"match": {"article_type" : "Editorial"}}, \
			{"match": {"article_type" : "Comment"}}, \
			{"match": {"article_type" : "Biography"}}
			]
	return filt

def get_concept_string(conceptid_series):
	result_string = ""
	for item in conceptid_series:
		result_string += item + " "
	
	return result_string.strip()

def get_query(full_conceptid_list, unmatched_terms, cursor):
	es_query = {}
	es_query["bool"] = { \
						"must_not": get_article_type_filters(), \
						"must": \
							[{"query_string": {"fields" : ["title_conceptids^5", "abstract_conceptids.*"], \
							 "query" : get_concept_query_string(full_conceptid_list, cursor)}}, \
							 {"query_string": {"fields" : ["article_title^5", "article_abstract.*"], \
							"query" : unmatched_terms}}]}
							
	return es_query


def get_concept_query_string(full_conceptid_list, cursor):
	query_string = ""
	for item in full_conceptid_list:
		if type(item) == list:
			query_string += "( "
			for concept in item:
				query_string += concept + " OR "
			query_string = query_string.rstrip('OR ')
			query_string += ") AND "
		else:
			query_string += item + " AND "
	query_string = query_string.rstrip("AND ")

	return query_string

def get_text_query(query):
	es_query = {"from" : 0, \
			 "size" : 20, \
			 "query": \
			 	{"bool": { \
					"must": \
						{"match": {"_all" : query}}, \
					"must_not": get_article_type_filters()}}}
	return es_query

def get_abstract_concept_arr_dict(concept_df):
	dict_arr = []
	for index,row in concept_df.iterrows():
		concept_dict = {'conceptid' : row['conceptid'], 'term' : row['term']}
		dict_arr.append(concept_dict)
	if len(dict_arr) == 0:

		return None
	else:
		return dict_arr

def get_query_arr_dict(query_concept_list):
	flattened_concept_df = pd.DataFrame()
	for item in query_concept_list:
		if type(item) == list:
			for concept in item:
				flattened_concept_df = flattened_concept_df.append(pd.DataFrame([[concept]], columns=['conceptid']))
		else:
			flattened_concept_df = flattened_concept_df.append(pd.DataFrame([[item]], columns=['conceptid']))
	flattened_concept_df = ann.add_names(flattened_concept_df)

	dict_arr = []
	for index,row in flattened_concept_df.iterrows():
		concept_dict = {'conceptid' : row['conceptid'], 'term' : row['term']}
		dict_arr.append(concept_dict)
	if len(dict_arr) == 0:
		return None
	else:
		return dict_arr

def check_concept_types_and_update_query(query_concept_list, unmatched_terms, cursor):
	concept_type_query_string = "select conceptid, \
		case when any(transitive_closure) = '404684003' \
		then 'symptom' \
		when any(transitive_closure) = '123037004' then 'anatomy' \
		when any(transitive_closure) = '363787002' then 'observable' \
		when any(transitive_closure) = '410607006' then 'cause' \
		when any(transitive_closure) = '373873005' then 'treatment' \
		when any(transitive_closure) = '71388002' then 'treatment' \
		when any(transitive_closure) = '105590001' then 'treatment' \
		when any(transitive_closure) = '362981000' then 'qualifier' end as concept_type \
		from annotation.transitive_closure"
	query_concept_type_df = pg.return_df_from_query(cursor, None, None, ["conceptid", "concept_type"])
	query_concept_list = query_concept_type_df['concept_type'].tolist()
	if unmatched_terms != "" and 'treatment' not in query_concept_list:
		if 'treatment' in unmatched_terms:
			treatment_conceptids_query = "select distinct (supertypeId) as conceptid from snomed.curr_transitive_closure_f where \
				subtypeId in ('373873005', '71388002', '105590001')"
			treatment_conceptids_df = pg.return_df_from_query(cursor, None, None, ["conceptid"])
			treatment_conceptids_list = treatment_conceptids_df["conceptid"].tolist()
			return query_concept_list.append(treatment_conceptids_list)
	else:
		return query_concept_list

