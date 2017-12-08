from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
import snomed_annotator as ann
import utilities.pglib as pg
from elasticsearch import Elasticsearch
from nltk.stem.wordnet import WordNetLemmatizer
import utilities.utils as u
import pandas as pd
import utilities.es_utilities as es_util
# Create your views here.

INDEX_NAME='pubmed3'

def home_page(request):
	return render(request, 'search/home_page.html')

### CONCEPT OVERRIDE FUNCTIONS
def concept_override(request):

	return render(request, 'search/concept_override.html')

def post_concept_override(request):
	cursor = pg.return_postgres_cursor()
	conceptid = request.POST['conceptid']
	description_id = request.POST['description_id']
	description = request.POST['description_text']

	if request.POST['action_type'] == 'add_description':
		u.add_description(conceptid, description, cursor)
	elif request.POST['action_type'] == 'deactivate_description_id':
		u.deactivate_description_id(description_id, cursor)
	elif request.POST['action_type'] == 'activate_description_id':
		u.activate_description_id(description_id, cursor)
	elif request.POST['action_type'] == 'add_concept':
		u.add_concept(description, cursor)

	cursor.close()
	return HttpResponseRedirect(reverse('search:concept_override'))

### ELASTIC SEARCH


def elastic_search_home_page(request):
	return render(request, 'search/elastic_home_page.html')


def post_elastic_search(request):
	query = request.POST['query']
	return HttpResponseRedirect(reverse('search:elastic_search_results', args=(query,)))

def elastic_search_results(request, query):
	es = Elasticsearch([{'host' : 'localhost', 'port' : 9200}])

	es_query = get_text_query(query)
	sr = es.search(index=INDEX_NAME, body=es_query)['hits']['hits']
	sr_payload = get_sr_payload(sr)

	return render(request, 'search/elastic_search_results_page.html', {'sr_payload' : sr_payload, 'query' : query})


### CONCEPT SEARCH

def concept_search_home_page(request):
	return render(request, 'search/concept_search_home_page.html')

def post_concept_search(request):
	query = request.POST['query']
	return HttpResponseRedirect(reverse('search:concept_search_results', args=(query,)))

def concept_search_results(request, query):
	es = Elasticsearch([{'host' : 'localhost', 'port' : 9200}])
	query = ann.clean_text(query)
	cursor = pg.return_postgres_cursor()
	filter_words_query = "select words from annotation.filter_words"
	filter_words_df = pg.return_df_from_query(cursor, filter_words_query, None, ["words"])
	query_concepts_df = ann.annotate_text_not_parallel(query, filter_words_df, cursor, True)
	

	sr = {}
	query_concepts_dict = None
	related_treatments_list = None
	if query_concepts_df is not None:
		unmatched_terms = get_unmatched_terms(query, query_concepts_df, filter_words_df)
		
		full_query_concepts_list = ann.query_expansion(query_concepts_df['conceptid'], cursor)

		related_df = get_related_conceptids(full_query_concepts_list, unmatched_terms, cursor)
		if related_df is not None:
			related_treatments_list = related_df['term'].tolist()


		query_concepts_dict = get_query_arr_dict(full_query_concepts_list)

		es_query = {"from" : 0, \
				 "size" : 20, \
				 "query": get_query(full_query_concepts_list, unmatched_terms, cursor)}
		sr = es.search(index=INDEX_NAME, body=es_query)

	else:
		es_query = get_text_query(query)
		sr = es.search(index=INDEX_NAME, body=es_query)

	sr_payload = get_sr_payload(sr['hits']['hits'])

	return render(request, 'search/concept_search_results_page.html', \
		{'sr_payload' : sr_payload, 'query' : query, 'concepts' : query_concepts_dict, \
		'at_a_glance' : {'treatments' : related_treatments_list}})




### Utility functions

def get_sr_payload(sr):
	sr_list = []

	for index,hit in enumerate(sr):
		hit_dict = {}
		sr_src = hit['_source']
		hit_dict['journal_title'] = sr_src['journal_title']
		hit_dict['pmid'] = sr_src['pmid']
		hit_dict['article_title'] = sr_src['article_title']
		hit_dict['journal_pub_year'] = sr_src['journal_pub_year']
		hit_dict['id'] = hit['_id']
		hit_dict['score'] = hit['_score']
		hit_dict = get_show_hide_components(sr_src, hit_dict)
		sr_list.append(hit_dict)
	return sr_list

def get_show_hide_components(sr_src, hit_dict):

	if sr_src['article_abstract'] is not None:
		top_key_list = list(sr_src['article_abstract'].keys())
		top_key = top_key_list[0]
		second_list = list(sr_src['article_abstract'][top_key].keys())
		second_key = second_list[0]
		hit_dict['abstract_show'] = (('abstract' if second_key == 'text' else second_key),
			sr_src['article_abstract'][top_key][second_key])
		hit_dict['abstract_hide'] = list()
		for key1,value1 in sr_src['article_abstract'].items():

			for key2,value2 in value1.items():
				if key1 == top_key and key2 == second_key:
					continue
				else:
					# print(hit_dict['abstract_hide'])
					hit_dict['abstract_hide'].append((key2, value2))
		hit_dict['abstract_hide'] = (None if len(hit_dict['abstract_hide']) == 0 else hit_dict['abstract_hide'])
	else:
		hit_dict['abstract_show'] = None
		hit_dict['abstract_hide'] = None
	return hit_dict

	# print(second_level_keys)

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

def get_concept_names_dict_for_sr(sr):

	concept_df = pd.DataFrame()
	c = u.Timer('start iteration')
	for hit in sr:
		try:
			if hit['_source']['title_conceptids'] is not None:
				concept_df = concept_df.append(pd.DataFrame(hit['_source']['title_conceptids'], columns=['conceptid']))
		except:
			pass
			# hit['_source']['title_conceptids'] = get_abstract_concept_arr_dict(title_df)
	

		abstract_concepts = es_util.get_flattened_abstract_concepts(hit)
		if abstract_concepts is not None:
			concept_df = concept_df.append(pd.DataFrame([[abstract_concepts]], columns=['conceptid']))

	c.stop()
	return sr
	# return concept_df



def get_article_type_filters():
	filt = [ \
			{"match": {"article_type" : "Letter"}}, \
			{"match": {"article_type" : "Editorial"}}, \
			{"match": {"article_type" : "Comment"}}, \
			{"match": {"article_type" : "Biography"}}, \
			{"match": {"article_type" : "Patient Education Handout"}}, \
			{"match": {"article_type" : "News"}}
			]
	return filt

def get_concept_string(conceptid_series):
	result_string = ""
	for item in conceptid_series:
		result_string += item + " "
	
	return result_string.strip()

def get_query(full_conceptid_list, unmatched_terms, cursor):
	es_query = {}

	if unmatched_terms is None:
		es_query["bool"] = { \
							"must_not": get_article_type_filters(), \
							"must": \
								[{"query_string": {"fields" : ["title_conceptids^5", "abstract_conceptids.*"], \
								 "query" : get_concept_query_string(full_conceptid_list, cursor)}}]}
	else:
		es_query["bool"] = { \
						"must_not": get_article_type_filters(), \
						"must": \
							[{"query_string": {"fields" : ["title_conceptids^5", "abstract_conceptids.*"], \
							 "query" : get_concept_query_string(full_conceptid_list, cursor)}}],
						"should": \
							[{"query_string": {"fields" : ["article_title^5", "article_abstract.*"], \
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

def get_related_conceptids(og_query_concept_list, unmatched_terms, cursor):
	# og_query_concept_list = og_query_concept_df['conceptid'].tolist()

	concept_type_query_string = "select * from annotation.concept_types where conceptid in %s"

	flattened_query_concept_list = [item for sublist in og_query_concept_list for item in sublist]
	# u.pprint(flattened_query_concept_list)
	query_concept_type_df = pg.return_df_from_query(cursor, concept_type_query_string, \
		(tuple(flattened_query_concept_list),), ["conceptid", "concept_type"])

	result_dict = dict()
	es = Elasticsearch([{'host' : 'localhost', 'port' : 9200}])

	for cid in og_query_concept_list:
		if isinstance(cid, list):	
			root_concept = u.get_conceptid_name(cid[0], cursor)

			if 'condition' in query_concept_type_df[query_concept_type_df['conceptid'].isin(cid)]['concept_type'].tolist():
				treatments_query = 	es_query = {"from" : 0, \
				 "size" : 400, \
				 "query": \
			 		{"bool": { \
						"must": \
							[{"query_string": {"fields" : ["title_conceptids", "abstract_conceptids"], \
							 "query" : get_concept_query_string([cid], cursor)}}], \
						"must_not": [get_article_type_filters(), {"query_string" : {"fields" : ["title_conceptids"],\
							"query" : '30207005'}}]}}}
				sr = es.search(index=INDEX_NAME, body=treatments_query)
				sr_conceptids = get_conceptids_from_sr(sr)
				if len(sr_conceptids) > 0:

					dist_sr_conceptids = list(set(sr_conceptids))

					sr_concept_type_query = """
							select
								conceptid 
								,concept_type
							from annotation.concept_types 
							where conceptid in %s and concept_type in ('treatment', 'diagnostic')
						"""
					agg_df = pg.return_df_from_query(cursor, sr_concept_type_query, (tuple(dist_sr_conceptids),), ["conceptid", "concept_type"])
					
					concept_types = list(set(agg_df['concept_type'].tolist()))
					result_dict[root_concept] = []
					
					for concept_type in concept_types:
						if concept_type == 'treatment':
							sr_conceptid_df = agg_df[agg_df['concept_type'] == concept_type].copy()
							sr_conceptid_df['count'] = 1
							sr_conceptid_df = sr_conceptid_df.groupby(['conceptid'], as_index=False)['count'].sum()

							if len(sr_conceptid_df) > 0:
								sr_conceptid_df = de_dupe_synonyms(sr_conceptid_df, cursor)
								sr_conceptid_df = ann.add_names(sr_conceptid_df)
								sr_conceptid_df = sr_conceptid_df.sort_values(['count'], ascending=False)
								
								result_dict[root_concept].append({'treatment' : sr_conceptid_df['term'].tolist()})
						
						if concept_type == 'diagnostic':
							sr_conceptid_df = agg_df[agg_df['concept_type'] == concept_type].copy()
							sr_conceptid_df['count'] = 1
							sr_conceptid_df = sr_conceptid_df.groupby(['conceptid'], as_index=False)['count'].sum()

							if len(sr_conceptid_df) > 0:
								sr_conceptid_df = de_dupe_synonyms(sr_conceptid_df, cursor)
								sr_conceptid_df = ann.add_names(sr_conceptid_df)
								sr_conceptid_df = sr_conceptid_df.sort_values(['count'], ascending=False)

								result_dict[root_concept].append({'diagnostic' : sr_conceptid_df['term'].tolist()})

							

			else:
				print('false')
		

	print(result_dict)
	return None

# this function isn't working - see schizophrenia treatment
def de_dupe_synonyms(df, cursor):

	synonyms = ann.get_concept_synonyms_df_from_series(df['conceptid'], cursor)

	for ind,t in df.iterrows():
		cnt = df[df['conceptid'] == t['conceptid']]
		ref = synonyms[synonyms['reference_conceptid'] == t['conceptid']]

		if len(ref) > 0:
			new_conceptid = ref.iloc[0]['synonym_conceptid']
			if len(df[df['conceptid'] == new_conceptid]):
				
				df.loc[ind, 'conceptid'] = new_conceptid

	df = df.groupby(['conceptid'], as_index=False)['count'].sum()

	return df 		

def get_conceptids_from_sr(sr):
	conceptid_list = []

	for hit in sr['hits']['hits']:
		if hit['_source']['title_conceptids'] is not None:
			conceptid_list.extend(hit['_source']['title_conceptids'])
		# if hit['_source']['abstract_conceptids'] is not None:
		# 	for key1 in hit['_source']['abstract_conceptids']:
		# 		for key2 in hit['_source']['abstract_conceptids'][key1]:
		# 			if hit['_source']['abstract_conceptids'][key1][key2] is not None:
		# 				conceptid_list.extend(hit['_source']['abstract_conceptids'][key1][key2])
	return conceptid_list


