from django.shortcuts import render
from django.views.decorators.csrf import requires_csrf_token
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from django.utils.safestring import mark_safe
import snomed_annotator2 as ann2
import utilities.pglib as pg
from nltk.stem.wordnet import WordNetLemmatizer
import utilities.utils2 as u
import pandas as pd
import utilities.es_utilities as es_util
from utilities.query import Query, get_pivot_queries
import math
from django.http import JsonResponse
import json
import numpy as np
import urllib.parse as urlparse
from urllib.parse import parse_qs
import nltk.data
import re


INDEX_NAME='pubmedx2.0'

# Initialize lemmatizer
lmtzr = WordNetLemmatizer()


### CONCEPT SEARCH


def concept_search_home_page(request):
	return render(request, 'search/concept_search_home_page.html', 
			{'sr_payload' : None, 'query' : '', 'query_annotation' : None, 'unmatched_list' : None, 'concepts' : None, 'primary_cids' : None,
				'journals': None, 'start_year' : '', 'end_year' : '', 'at_a_glance' : {'related' : None}, \
				'treatment' : None, 'diagnostic' : None, 'cause' : None, 'condition' : None, \
				'calcs' : None}
			)


def get_query_filters(data):
	filters = {}
	try:
		filters['journals'] = []
		for item in data['journals']:
			filters['journals'].append(item["tag"])
		
	except:
		filters['journals'] = []

	try:
		if data['start_year'] == '':
			filters['start_year'] = ''
		else:
			filters['start_year'] = int(data['start_year'])
	except:
		filters['start_year'] = ''

	try:
		if data['end_year'] == '':
			filters['end_year'] = '' 
		else:
			filters['end_year'] = int(data['end_year'])
	except:
		filters['end_year'] = ''

	return filters

def return_section_sentences(text, section, section_index, sentences_df):
	text = text.replace('...', '.')
	text = text.replace('. . .', '.')
	tokenized = nltk.sent_tokenize(text)

	for ln_num, line in enumerate(tokenized):
		sentences_df = sentences_df.append(pd.DataFrame([[line, section, section_index, ln_num]],
			columns=['line', 'section', 'section_ind', 'ln_num']), sort=False)

	return sentences_df

def about(request):
	return render(request, 'about/about.html')

def terms(request):
	return render(request, 'about/terms.html')
	
@requires_csrf_token
def post_search_text(request):
	c = {}
	conn, cursor = pg.return_postgres_cursor()
	es = es_util.get_es_client()
	pivot_cid = []
	unmatched_list = []
	pivot_term = None
	query_type = ''
	primary_a_cids = []
	query = ''
	filters = {}
	query = None
	spellcheck_threshold = 100

	# If GET request, this is from a link
	if request.method == 'GET': 
		parsed = urlparse.urlparse(request.path)
		parsed = parse_qs(parsed.path)

		query = parsed['/search/query'][0]

		if 'primary_a_cids[]' in parsed:
			primary_a_cids = parsed['primary_a_cids[]']

		if 'pivot_complete_acid[]' in parsed:
			pivot_complete_acid = parsed['pivot_complete_acid[]']

		if 'pivot_history[]' in parsed:
			pivot_history = parsed['pivot_history[]']

		if 'unmatched_list[]' in parsed:
			unmatched_list= parsed['unmatched_list[]']
		
		if 'pivot_cid' in parsed:
			primary_cids.extend(parsed['pivot_cid'])

		if 'start_year' in parsed:
			filters['start_year'] = int(parsed['start_year'][0])
		else:
			filters['start_year'] = ''

		if 'end_year' in parsed:
			filters['end_year'] = int(parsed['end_year'][0])
		else:
			filters['end_year'] = ''

		filters['journals'] = []
		for key,value in parsed.items():
			if key.startswith('journals['):
				filters['journals'].append(value[0])

		if 'query_type' in parsed:
			query_type = parsed['query_type'][0]

	if request.method == 'POST':

		data = json.loads(request.body)

		query = data['query']

		filters = get_query_filters(data)
		query_type = data['query_type']

		if 'unmatched_list' in data:
			unmatched_list = data['unmatched_list']

		else: 
			unmatched_list = []

		if 'primary_a_cids' in data:
			primary_a_cids = data['primary_a_cids']
		else:
			primary_a_cids = []

		if 'pivot_complete_acid' in data:
			pivot_complete_acid = data['pivot_complete_acid']
		else:
			pivot_complete_acid = []

		if data['pivot_cid'] is not None:
			pivot_cid = [data['pivot_cid']]
			# if data['pivot_cid'] is not None:
			primary_a_cids.append([data['pivot_cid']])
		else:
			pivot_cid = []

		if 'pivot_history' in data:
			pivot_history = data['pivot_history']
			pivot_history.extend(pivot_cid)
		else:
			pivot_history = pivot_cid

	sr = dict()
	related_dict = {}
	treatment_dict = {}
	condition_dict = {}
	diagnostic_dict = {}
	cause_dict = {}
	calcs_json = {}

	flattened_query = None

	if query_type == 'pivot':
		flattened_concepts_list = [a_cid for item in primary_a_cids for a_cid in item]
		query_types_list = get_query_concept_types_df(flattened_concepts_list, cursor)['concept_type'].tolist()
		full_query_concepts_list = ann2.query_expansion(primary_a_cids, flattened_concepts_list, None, cursor)
		pivot_term = None

		if request.method == 'POST':
			pivot_term = data['pivot_term']
			query = query + ' ' + pivot_term
		else:
			query = query + ' ' + parsed['pivot_term'][0]

		params = filters 

		flattened_query = get_flattened_query_concept_list(full_query_concepts_list)
		root_query = get_query(full_query_concepts_list, unmatched_list, query_types_list, filters['journals'], 
					filters['start_year'], filters['end_year'], 
				 	["title_cids^10", "abstract_conceptids.*"], cursor)

		query_obj = Query(full_query_concepts_list = full_query_concepts_list, 
				flattened_concept_list = flattened_concepts_list,
				flattened_query = flattened_query,
				query_types_list = query_types_list,
				unmatched_list = unmatched_list,
				filters = filters,
				root_query = root_query,
				pivot_history = pivot_history
				)

		es_query = {"from" : 0, \
				 "size" : 20, \
				 "query": query_obj.root_query}

		sr = es.search(index=INDEX_NAME, body=es_query, request_timeout=100000)

		treatment_dict, diagnostic_dict, condition_dict, cause_dict, narrowed_query_a_cids = get_related_conceptids(query_obj, cursor)

		sr_payload = get_sr_payload(sr['hits']['hits'], narrowed_query_a_cids, unmatched_list, cursor)

	elif query_type == 'keyword':

		if len(query.split(' ')) == 1:
			raw_query = """
				select acid from annotation2.lemmas t1 
				join annotation2.concept_counts t2 on t1.acid=t2.concept where term_lower=%s
				order by cnt desc limit 1 
			"""
		else:
			raw_query = "select acid from annotation2.lemmas where term_lower=%s"

		query_concepts_df = pg.return_df_from_query(cursor, raw_query, (query.lower(),), ["acid"])

		if (len(query_concepts_df.index) == 1):
			query_concepts_df['term_start_index'] = 0
			query_concepts_df['term_end_index'] = len(query.split(' '))-1

		elif (len(query_concepts_df.index) != 1):
			query = ann2.clean_text(query)
			words = ann2.get_all_words_list(query)

			# filter out special characters
			words = [word for word in words if word.isalnum()]

			cache = ann2.get_cache(all_words_list=words, case_sensitive=False, \
			check_pos=False, spellcheck_threshold=spellcheck_threshold, lmtzr=lmtzr)

			query_df = return_section_sentences(query, 'query', 0, pd.DataFrame())
			query_concepts_df = ann2.annotate_text_not_parallel(sentences_df=query_df, cache=cache, \
			case_sensitive=False, check_pos=False, bool_acr_check=False,\
			spellcheck_threshold=spellcheck_threshold, \
			write_sentences=False, lmtzr=lmtzr)
	
		primary_a_cids = []

		if not query_concepts_df['acid'].isnull().all():
			query_concepts_df = query_concepts_df[query_concepts_df['acid'].notna()].drop_duplicates(subset=['acid']).copy()

			term_indices = set(query_concepts_df['term_start_index'].tolist())

			for index in term_indices:
				primary_a_cids.append(query_concepts_df[query_concepts_df['term_start_index']==index]['acid'].tolist())

			flattened_concepts_list = [a_cid for item in primary_a_cids for a_cid in item]

			# in the event of keyword search after pivot
			# pivot history will be out of sync (ex deleting pivot term)
			pivot_history = list(set(pivot_history).intersection(set(flattened_concepts_list)))
			
			query_types_list = get_query_concept_types_df(flattened_concepts_list, cursor)['concept_type'].tolist()

			unmatched_list = get_unmatched_list(query, query_concepts_df)

			# full_query_concepts_list format = [[[], []], []]
			full_query_concepts_list = ann2.query_expansion(primary_a_cids, \
				flattened_concepts_list, None, cursor)

			flattened_query = get_flattened_query_concept_list(full_query_concepts_list)

			query_concept_count = len(query_concepts_df.index)

			
			root_query = get_query(full_query_concepts_list, unmatched_list, query_types_list \
						 	,filters['journals'], filters['start_year'], filters['end_year'] \
						 	,["title_cids^10", "abstract_conceptids.*"], cursor)

			query_obj = Query(full_query_concepts_list = full_query_concepts_list, 
				flattened_concept_list = flattened_concepts_list,
				flattened_query = flattened_query,
				query_types_list = query_types_list,
				unmatched_list = unmatched_list,
				filters = filters,
				root_query = root_query,
				pivot_history = pivot_history
				)
				
			sr = es.search(index=INDEX_NAME, body= {"from" : 0, "size" : 20, "query": query_obj.root_query}, request_timeout=100000)	

		if query_concepts_df['acid'].isnull().all() or len(sr['hits']['hits']) == 0:
			#query filtered for non-alphanumeric standalone characters
			unmatched_list = words
			root_query = get_keyword_query(query, filters['journals'], filters['start_year'], filters['end_year'])

			query_obj = Query(unmatch_list = words, root_query = root_query, pivot_history=pivot_history)
			sr = es.search(index=INDEX_NAME, body={"from" : 0, "size" : 20, "query": query_obj.root_query})

		treatment_dict, diagnostic_dict, condition_dict, cause_dict, narrowed_query_a_cids = get_related_conceptids(query_obj, cursor)

		sr_payload = get_sr_payload(sr['hits']['hits'], narrowed_query_a_cids, unmatched_list, cursor)


	if query_obj.flattened_concept_list is not None:
		calcs_json = get_calcs(query_obj.flattened_concept_list, cursor)
	else:
		calcs_json = {}
	ip = get_ip_address(request)
	log_query(ip, query, primary_a_cids, unmatched_list, filters, cursor)
	cursor.close()
	conn.close()

	if request.method == 'POST':
		html = render(request, 'search/concept_search_results_page.html', {'sr_payload' : sr_payload, 
				'query' : query,
				'primary_a_cids' : primary_a_cids,
				'unmatched_list' : unmatched_list,
				'pivot_history' : pivot_history,
				'journals': filters['journals'], 'start_year' : filters['start_year'], 'end_year' : filters['end_year'], \
				'treatment' : treatment_dict, 'diagnostic' : diagnostic_dict, 'cause' : cause_dict, 'condition' : condition_dict, \
				'calcs' : calcs_json})
		return html

	elif request.method == 'GET':
		return render(request, 'search/concept_search_home_page.html', {'sr_payload' : sr_payload, 'query' : query, 
				'query_annotation' : query_concepts_df['acid'].tolist(),
				'narrowed_query_a_cids' : expanded_query_acids,
				'unmatched_list' : unmatched_list, 
				'pivot_history' : pivot_history,
				'journals': filters['journals'], 
				'start_year' : filters['start_year'], 'end_year' : filters['end_year'],
				'treatment' : treatment_dict, 'diagnostic' : diagnostic_dict, 'cause' : cause_dict, 'condition' : condition_dict,
				'calcs' : calcs_json}, c)


def get_ip_address(request):
	ip = ''
	x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
	if x_forwarded_for:
		ip = x_forwarded_for.split(',')[0]
	else:
		ip = request.META.get('REMOTE_ADDR')
	return ip

def get_calcs(concepts, cursor):
	if len(concepts) > 0:
		query = """
			select distinct on (title, t1.description, url) title, t1.description, url 
			from annotation2.mdc_final t1 where acid in %s 
		"""
		calcs = pg.return_df_from_query(cursor, query, (tuple(concepts),), ['title', 'desc', 'url'])

		calc_json = []
		calcs_dict = calcs.to_dict('records')
		for row in calcs_dict:
			calc_json.append({'title' : row['title'], 'desc' : row['desc'], 'url' : row['url']})
		return calc_json
	else:
		return None

def rollups(cids_df, cursor):
	if len(cids_df.index) > 0:
		params = (tuple(cids_df['acid']), tuple(cids_df['acid']), tuple(cids_df['acid']), tuple(cids_df['acid']))
		query = """
			select
				child_acid
				,parent_acid
			from snomed2.transitive_closure_acid
			where child_acid in %s and parent_acid in %s 
			and parent_acid not in 
			(select child_acid from snomed2.transitive_closure_acid 
			where child_acid in %s and parent_acid in %s)
		"""
		parents_df = pg.return_df_from_query(cursor, query, params, ["child_acid", "parent_acid"])

		parents_df = parents_df[parents_df['parent_acid'].isin(parents_df['child_acid'].tolist()) == False].copy()

		orphan_df = cids_df[(cids_df['acid'].isin(parents_df['child_acid'].tolist()) == False) 
			& (cids_df['acid'].isin(parents_df['parent_acid'].tolist()) == False)]
		orphan_df = orphan_df.sort_values(by=['count'], ascending=False)

		json = []

		joined_df = cids_df.merge(parents_df, how='right', left_on='acid', right_on='child_acid')

		distinct_parents = parents_df.drop_duplicates(['parent_acid'])
		
		parents_count = joined_df.groupby(['parent_acid'])['count'].sum().reset_index()
		
		distinct_parents = distinct_parents.merge(parents_count, how='left', left_on='parent_acid', right_on='parent_acid')
		distinct_parents = distinct_parents.sort_values(by=['count'], ascending=False)

		distinct_parents = distinct_parents['parent_acid'].tolist()

		assigned_child_acid = []
		for parent in distinct_parents:
			children_df = joined_df[(joined_df['parent_acid'] == parent) & (~joined_df['child_acid'].isin(assigned_child_acid))]
			count = children_df['count'].sum()
			count = int(count + cids_df[cids_df['acid'] == parent]['count'].values[0])
			parent_name = cids_df[cids_df['acid'] == parent]['term'].values[0]
			children_name_list = children_df['term'].tolist()
			assigned_child_acid.extend(children_df['child_acid'].tolist())
			complete_acid_list = children_df['child_acid'].tolist()
			complete_acid_list.append(parent)
			parent_dict = {parent : {'name' : parent_name, 'count' : count, 
				'children' : children_name_list, 'complete_acid' : complete_acid_list}}
			json.append(parent_dict)

		orphan_dict = orphan_df.to_dict('records')
		for row in orphan_dict:
			r = {row['acid'] : {'name' : row['term'], 'count' : row['count'], 
				'children' : [], 'complete_acid' : [row['acid']]}}
			json.append(r)

		return json
	else:
		return None

# Takes sorted inputs, which is not ideal
def get_json(item_df, prev_json):

	if item_df['parent_count'] == 0:
		prev_json.append({item_df['acid'] : {'name' : item_df['term'], 'count' : item_df['count'], 'children' : []}})
		return prev_json
	else:
		for i,r in enumerate(prev_json):
			if item_df['parent_acid'] in r.keys():
				r[item_df['parent_acid']]['children'].append({item_df['acid'] : {'name' : item_df['term'], 'count' : item_df['count'], 'children' : []}})
				return prev_json
		for i,r in enumerate(prev_json):
			get_json(item_df, r[list(r.keys())[0]]['children'])
			return prev_json

def log_query (ip_address, query, primary_cids, unmatched_list, filters, cursor):
	insert_query = """
		insert into search.query_logs
		VALUES(%s, %s, %s, %s, %s,%s, %s, now())
	"""
	cursor.execute(insert_query, (ip_address, query,json.dumps(primary_cids), \
		unmatched_list,filters['start_year'], filters['end_year'], json.dumps(filters['journals'])))

	cursor.connection.commit()



### Utility functions

def get_sr_payload(sr, expanded_query_acids, unmatched_list, cursor):
	sr_list = []
	expanded_query_acids = [a_cid for item in expanded_query_acids for a_cid in item]

	terms = []
	if len(expanded_query_acids) > 0:
		query = """
			select term from annotation2.used_descriptions where acid in %s order by length(term) desc
		"""
		terms = pg.return_df_from_query(cursor, query, (tuple(expanded_query_acids),), ["term"])["term"].tolist()

	terms.extend(unmatched_list)

	for index,hit in enumerate(sr):
		hit_dict = {}
		sr_src = hit['_source']
		for term in terms:
			term_search = r"(\b" + term + r"s?\b)(?!(.(?!<b))*</b>)"

			if sr_src['article_abstract'] is not None:
				for key in sr_src['article_abstract']:
					if term.upper() == term:
						sr_src['article_abstract'][key] = re.sub(term_search, r'<b>\1</b>', sr_src['article_abstract'][key])
					else:
						sr_src['article_abstract'][key] = re.sub(term_search, r'<b>\1</b>', sr_src['article_abstract'][key], \
							flags=re.IGNORECASE)
					sr_src['article_abstract'][key] = mark_safe(sr_src['article_abstract'][key])

			if term.upper() == term:
				sr_src['article_title'] = re.sub(term_search, r'<b>\1</b>', sr_src['article_title'])
			else:
				sr_src['article_title'] = re.sub(term_search, r'<b>\1</b>', sr_src['article_title'], flags=re.IGNORECASE)
			sr_src['article_title'] = mark_safe(sr_src['article_title'])

		hit_dict['journal_title'] = sr_src['journal_iso_abbrev']
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

		hit_dict['abstract_show'] = (('abstract' if top_key == 'unassigned' else top_key),
			sr_src['article_abstract'][top_key])
		hit_dict['abstract_hide'] = list()
		for key1,value1 in sr_src['article_abstract'].items():
		
			if key1 == top_key:
				continue
			else:
				hit_dict['abstract_hide'].append((key1, value1))
		hit_dict['abstract_hide'] = (None if len(hit_dict['abstract_hide']) == 0 else hit_dict['abstract_hide'])
	else:
		hit_dict['abstract_show'] = None
		hit_dict['abstract_hide'] = None
	return hit_dict


def get_unmatched_list(query, query_concepts_df):
	unmatched_list = []
	for index,word in enumerate(query.split()):
		if not query_concepts_df.empty and len((query_concepts_df[(query_concepts_df['term_end_index'] >= index) \
			& (query_concepts_df['term_start_index'] <= index)]).index) > 0:
			continue
		else:
			unmatched_list.append(word)
	return unmatched_list




def get_article_type_filters():
	filt = [ \
			{"term": {"article_type" : "Letter"}}, \
			{"term": {"article_type" : "Editorial"}}, \
			{"term": {"article_type" : "Comment"}}, \
			{"term": {"article_type" : "Biography"}}, \
			{"term": {"article_type" : "Patient Education Handout"}}, \
			{"term": {"article_type" : "News"}},
			{"term": {'article_title': "rat"}},
			{"term": {'article_title': "mice"}},
			{"term": {'article_title': "mouse"}},
			{"term": {'article_title': "dog"}},
			{"term": {'article_title': "dogs"}},
			{"term": {'article_title': "cat"}},
			{"term": {'article_title': "cats"}},
			{"term": {'article_title': "rabbit"}},
			{"term": {'article_title': "rabbits"}},
			{"term": {'article_title': "guinea-pig"}},
			{"term": {'article_title': "bovine"}},
			{"term": {'article_title': "postpartum"}},
			{"term": {'article_title': "guinea pig"}},
			{"term": {'article_title': "rats"}},
			{"term": {'article_title': "murine"}},
			]
	return filt

def get_concept_query_string(full_conceptid_list):
	query_string = "( "
	for a_cid_list in full_conceptid_list:
		query_string += "( "
		for item in a_cid_list:
			query_string += "("

			counter = 10
			
			for concept in item:
				query_string += concept + "^" + str(counter) + " OR "
				if counter != 1:
					counter -= 1
			query_string = query_string.rstrip('OR ')
			query_string += ") OR "

		query_string = query_string.rstrip(" OR ")

		query_string += ") AND " 

	query_string = query_string.rstrip("AND ")
	query_string += ") "
	return query_string

def get_article_type_query_string(concept_type_list, unmatched_list):
	# RCT, meta-analysis, network-meta, cross-over study, case report
	if 'condition' in concept_type_list and ('treatment' in concept_type_list or 'treatment' in unmatched_list):
		return "( 887729^15 OR 887761^7 OR 887749^7 OR 887770^2 OR 887774^2 OR 887763^1)"
	# prevention procedure not in concept_type
	# Systematic review, meta-analysis
	elif 'condition' in concept_type_list and len(unmatched_list) == 0 and \
		 '379416' not in concept_type_list and 'symptom' not in concept_type_list:
		return "( 887764^10 OR 887761^8 )"
	# Cohort study, systematic review, case-control study, RCT
	elif 'symptom' in concept_type_list:
		return "( 887774^10 OR  887764^8 OR 887765^5 OR 887729^0.5 )"
	# RCT, Network meta, meta-analysis, systematic review
	elif 'treatment' in concept_type_list:
		return "( 887729^10 OR 887749^8 OR 887761^10 OR 887764^8 )"
	# Case-control study, case report
	elif 'anatomy' in concept_type_list:
		return "( 887765^5 OR 887763^2 )"
	# Preventive procedure in concept_type_list
	# RCT, Longitudinal study, cohort study, cost-benefit analysis
	elif 'prevention' in concept_type_list or 'prevention' in unmatched_list or '379416' in concept_type_list:
		return "( 887729^10 OR 887772^5 OR 887774^5 OR 887769^5 )"
	else:
		return ""


def get_query(full_conceptid_list, unmatched_list, query_types_list, journals, start_year, end_year, fields_arr, cursor):
	es_query = {}
	if len(unmatched_list) == 0:

		es_query["function_score"] = {"query" : { "bool" : {
							"must_not": get_article_type_filters(), 
							"must": 
								[{"query_string": {"fields" : fields_arr, 
								 "query" : get_concept_query_string(full_conceptid_list)}}], 
							"should": 
								[{"query_string" : {"fields" : fields_arr, 
								"query" : get_article_type_query_string(query_types_list, unmatched_list)}}]}}, 
							"functions" : []}
	else:
		# Unmatched terms need to be formatted for lucene
		es_query["function_score"] = {"query" : {"bool" : {
						"must_not": get_article_type_filters(), 
						"must": \
							[{"query_string": {"fields" : fields_arr, 
							 	"query" : get_concept_query_string(full_conceptid_list)}}, 
							 	{"query_string": { "fields" : ["article_abstract.*"],
								"query" : get_unmatched_query_string(unmatched_list)}}
							 	],
						"should": 
							[{"query_string" : {"fields" : fields_arr, 
							"query" : get_article_type_query_string(query_types_list, unmatched_list)}}
							]
						}}, 
						"functions" : []
						}
	
	if (len(journals) > 0) or start_year or end_year:
		d = []

		if len(journals) > 0:

			for i in journals:
				d.append({"term" : {'journal_iso_abbrev' : i}})

		if start_year and end_year:
			d.append({"range" : {"journal_pub_year" : {"gte" : start_year, "lte" : end_year}}})
		elif start_year:
			d.append({"range" : {"journal_pub_year": {"gte" : start_year}}})
		elif end_year:
			d.append({"range" : {"journal_pub_year" : {"lte" : end_year}}})

		es_query["function_score"]["query"]["bool"]["filter"] = d

	if not start_year:
		es_query["function_score"]["functions"].append(\
			{"filter" : {"range": {"journal_pub_year": {"gte" : "1990"}}}, "weight" : 1.4})

	return es_query

def get_es_query_filters(journals, start_year, end_year):
	filters = []
	if (len(journals) > 0) or start_year or end_year:
		if len(journals) > 0:
			for i in journals:
				filters.append({"term" : {'journal_iso_abbrev' : i}})

		if start_year and end_year:
			filters.append({"range" : {"journal_pub_year" : {"gte" : start_year, "lte" : end_year}}})
		elif start_year:
			filters.append({"range" : {"journal_pub_year" : {"gte" : start_year}}})
		elif end_year:
			filters.append({"range" : {"journal_pub_year" : {"lte" : end_year}}})

	return filters


def get_keyword_query(query, journals, start_year, end_year):
	es_query = {
				"function_score" : {
					 "query": \
					 	{"bool": { \
							"must": \
								{"query_string": {"query" : query}}, \
							"must_not": get_article_type_filters()}},
						"functions" : [{"filter" : {"range": {"journal_pub_year": {"gte" : "1990"}}}, "weight" : 1.4}]
				}
			}
	filters = get_es_query_filters(journals, start_year, end_year)
	if filters:
		es_query["function_score"]["query"]["bool"]["filter"] = filters

	return es_query


def get_unmatched_query_string(unmatched_list):
	query_string = ""
	for i in unmatched_list:
		query_string += "( " + i + " ) AND "
	query_string = query_string.rstrip("AND ")

	return query_string


def get_flattened_query_concept_list(concept_list):
	flattened_query_concept_list = list()

	for i in concept_list:
		if isinstance(i, list):
			for g in i:
				flattened_query_concept_list.append(g)
		else:
			flattened_query_concept_list.append(i)
	return flattened_query_concept_list

def get_query_concept_types_df(flattened_concept_list, cursor):
	concept_type_query_string = """
		select root_acid as acid, rel_type as concept_type 
		from annotation2.concept_types where active=1 and root_acid in %s
	"""
	query_concept_type_df = pg.return_df_from_query(cursor, concept_type_query_string, \
		(tuple(flattened_concept_list),), ["acid", "concept_type"])
	return query_concept_type_df

def get_narrowed_pivot_types_df(pivot_history, cursor):
	pivot_values_format = '(' + '), ('.join(f'\'{item}\'' for item in pivot_history) + ')'
	pivot_list_format = '(' + ', '.join(f'\'{item}\'' for item in pivot_history) + ')'
	concept_type_query_string = """
		select 
			parent_acid
			,rel_type
		from (
			select 
				parent_acid
			from 
				(
					select 
						parent_acid from (VALUES %s) as t (parent_acid)
				) t1
			where parent_acid not in 
				(select parent_acid from (
					select * from snomed2.transitive_closure_acid 
					where child_acid in %s
					) t2 
					where parent_acid in %s
				)
			) t3
		join annotation2.concept_types t4
		on t3.parent_acid = t4.root_acid
		where active=1
	""" % (pivot_values_format, pivot_list_format, pivot_list_format)

	query_concept_type_df = pg.return_df_from_query(cursor, concept_type_query_string, \
		None, ["acid", "concept_type"])
	return query_concept_type_df



def get_query_concept_types_df_3(conceptid_df, query_obj, cursor):
	dist_concept_list = list(set(conceptid_df['acid'].tolist()))
	if len(dist_concept_list) > 0:
		if query_obj.flattened_query is not None:
			if len(query_obj.pivot_history) == 0:
				query_concept_list = [a_cid for item in query_obj.flattened_query for a_cid in item]
				concept_type_query_string = """
					select 
						root_acid as acid
						,rel_type as concept_type
					from annotation2.concept_types
					where active=1 and 
					root_acid not in (select treatment_acid from ml2.labelled_treatments where label=2)
					and root_acid in %s and rel_type != 'treatment' and rel_type != 'symptom'
					
					union
					
					select distinct(treatment_acid) as acid
						,'treatment' as concept_type
					from ml2.treatment_recs_final_1
					where condition_acid in %s and treatment_acid in %s 
					and treatment_acid in
						(select root_acid from annotation2.concept_types where active=1 and rel_type='treatment')
				"""
				query_concept_type_df = pg.return_df_from_query(cursor, concept_type_query_string, \
					(tuple(dist_concept_list), tuple(query_concept_list), tuple(dist_concept_list)), ["acid", "concept_type"])

				conceptid_df = pd.merge(conceptid_df, query_concept_type_df, how='inner', on=['acid'])
			else:
				query_concept_list = [a_cid for item in query_obj.flattened_query for a_cid in item]
				pivot_types = get_narrowed_pivot_types_df(query_obj.pivot_history, cursor)
				related_df = pd.DataFrame()

				pivot_condition = pivot_types[pivot_types['concept_type'] == 'condition']['acid'].tolist()
				pivot_treatment = pivot_types[pivot_types['concept_type'] == 'treatment']['acid'].tolist()
				pivot_diagnostic = pivot_types[pivot_types['concept_type'] == 'diagnostic']['acid'].tolist()
				pivot_cause = pivot_types[pivot_types['concept_type'] == 'cause']['acid'].tolist()

				pivot_query = get_pivot_queries('condition', dist_concept_list, query_concept_list, pivot_condition, cursor)
				if pivot_query is not None:
					related_df = related_df.append(pg.return_df_from_query(cursor, pivot_query, None, ["acid", "concept_type"]))


				pivot_query = get_pivot_queries('treatment', dist_concept_list, query_concept_list, pivot_treatment, cursor)
				if pivot_query is not None:
					related_df = related_df.append(pg.return_df_from_query(cursor, pivot_query, None, ["acid", "concept_type"]))

				pivot_query = get_pivot_queries('diagnostic', dist_concept_list, query_concept_list, pivot_diagnostic, cursor)

				if pivot_query is not None:
					related_df = related_df.append(pg.return_df_from_query(cursor, pivot_query, None, ["acid", "concept_type"]))

				pivot_query = get_pivot_queries('cause', dist_concept_list, query_concept_list, pivot_cause, cursor)
				if pivot_query is not None:
					related_df = related_df.append(pg.return_df_from_query(cursor, pivot_query, None, ["acid", "concept_type"]))

				conceptid_df = pd.merge(conceptid_df, related_df, how='inner', on=['acid'])
				
			return conceptid_df
		# Case for keyword search
		else:
			concept_type_query_string = """
				select 
					root_acid as acid
					,rel_type as concept_type
				from annotation2.concept_types
				where active=1 and 
				root_acid not in (select treatment_acid from ml2.labelled_treatments where label=2)
				and rel_type != 'treatment' and rel_type != 'symptom'
				
				union
				
				select distinct(treatment_acid) as acid
					,'treatment' as concept_type
				from ml2.treatment_recs_final_1
				where treatment_acid in
					(select root_acid from annotation2.concept_types where active=1 and rel_type='treatment')
			"""
			query_concept_type_df = pg.return_df_from_query(cursor, concept_type_query_string, \
				None, ["acid", "concept_type"])

			conceptid_df = pd.merge(conceptid_df, query_concept_type_df, how='inner', on=['acid'])
			return conceptid_df
		
	else:
		concept_type_query_string = """
			select root_cid as acid, rel_type as concept_type
			from annotation2.concept_types
			where active=1 and rel_type='condition' or rel_type='cause'

		"""
		query_concept_type_df = pg.return_df_from_query(cursor, concept_type_query_string, None, ["acid", "concept_type"])
		conceptid_df = pd.merge(conceptid_df, query_concept_type_df, how='right', on=['acid'])
		return conceptid_df



# query_concept_list, primary_cids, flattened_query, query_types_list, unmatched_list, filters, 
def get_related_conceptids(query_obj, cursor):
	result_dict = dict()
	es = es_util.get_es_client()

	# size zero means only aggregation results returned

	es_query = {
				 "size" : 0, \
				 "query": query_obj.root_query, \
				 "aggs" : {'concepts_of_interest' : {"terms" : {"field" : "concepts_of_interest", "size" : 40000}}, 
			}}
	sr = es.search(index=INDEX_NAME, body=es_query, request_timeout=100000)

	# try:
	res = sr['aggregations']['concepts_of_interest']['buckets']
	cids_of_interest_df = pd.DataFrame.from_dict(res)
	cids_of_interest_df.columns = ['acid', 'count']

	# except:
	# 	cids_of_interest_df = pd.DataFrame()

	narrowed_query_a_cids = []
	
	if len(cids_of_interest_df) > 0:
		if query_obj.flattened_query is not None:
			narrowed_query_a_cids = cids_of_interest_df[cids_of_interest_df['acid'].isin(query_obj.flattened_query)].copy()['acid'].tolist()
			cids_of_interest_df = cids_of_interest_df[cids_of_interest_df['acid'].isin(query_obj.flattened_concept_list) == False].copy()
		else:
			narrowed_query_a_cids = cids_of_interest_df['acid'].tolist()
			cids_of_interest_df = cids_of_interest_df.copy()
	
	if len(narrowed_query_a_cids) == 0:
		narrowed_query_a_cids = query_obj.flattened_query

	sub_dict = dict()
	sub_dict['treatment'] = []
	sub_dict['diagnostic'] = []
	sub_dict['condition'] = []
	sub_dict['cause'] = []

	if len(cids_of_interest_df.index) > 0:
		concept_types_df = get_query_concept_types_df_3(cids_of_interest_df, query_obj, cursor)

		if len(concept_types_df.index) > 0:
			concept_types_df = ann2.add_names(concept_types_df, cursor)
			agg_tx = concept_types_df[concept_types_df['concept_type'] == 'treatment']
			if len(agg_tx.index) > 0:
				sub_dict['treatment'] = rollups(agg_tx, cursor)

			agg_dx = concept_types_df[concept_types_df['concept_type'] == 'diagnostic']
			if len(agg_dx.index) > 0:
				sub_dict['diagnostic'] = rollups(agg_dx, cursor)
				
			agg_cz = concept_types_df[concept_types_df['concept_type'] == 'cause']	
			if len(agg_cz.index) > 0:
				sub_dict['cause'] = rollups(agg_cz, cursor)


			agg_condition = concept_types_df[concept_types_df['concept_type'] == 'condition']
			if len(agg_condition) > 0:
				sub_dict['condition'] = rollups(agg_condition, cursor)

	return sub_dict['treatment'], sub_dict['diagnostic'], sub_dict['condition'], sub_dict['cause'], narrowed_query_a_cids
	


def get_conceptids_from_sr(sr):
	conceptid_list = []

	for hit in sr['hits']['hits']:
		if hit['_source']['title_cids'] is not None:
			conceptid_list.extend(hit['_source']['title_cids'])
		if hit['_source']['abstract_conceptids'] is not None:
			for key1 in hit['_source']['abstract_conceptids']:
				if hit['_source']['abstract_conceptids'][key1] is not None:
					conceptid_list.extend(hit['_source']['abstract_conceptids'][key1])
	return conceptid_list

def get_title_cids(sr):

	conceptid_df = pd.DataFrame(columns=['acid', 'pmid'])
	for hit in sr['hits']['hits']:
		if hit['_source']['title_cids'] is not None:
			pmid = hit['_source']['pmid']
			cid_list = list(set(hit['_source']['title_cids']))
			if 'conclusions_cid' in hit['_source']:
				cid_list.append(list(set(hit['_source']['conclusions_cid'])))
			new_df = pd.DataFrame()
			new_df['acid'] = cid_list
			new_df['pmid'] = pmid
			conceptid_df = conceptid_df.append(new_df)
	return conceptid_df

def get_sr_pmids(sr):
	pmids = []
	for hit in sr['hits']['hits']:
		pmid = hit['_source']['pmid']
		if pmid is not None:
			pmids.append(pmid)
	return pmids


def get_abstract_cids(sr):
	conceptid_df = pd.DataFrame(columns=['conceptid', 'pmid'])
	for hit in sr['hits']['hits']:
		if hit['_source']['abstract_conceptids'] is not None:
			pmid = hit['_source']['pmid']
			for key1 in hit['_source']['abstract_conceptids']:
				if hit['_source']['abstract_conceptids'][key1] is not None:
					for cid in list(set(hit['_source']['abstract_conceptids'][key1])):
						conceptid_df = conceptid_df.append(pd.DataFrame([[cid, pmid]], columns=['conceptid', 'pmid']))
	return conceptid_df
