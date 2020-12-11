from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
import snomed_annotator2 as ann2
import utilities.pglib as pg
from nltk.stem.wordnet import WordNetLemmatizer
import utilities.utils2 as u2
import pandas as pd
import utilities.es_utilities as es_util
import math
from django.http import JsonResponse
import json
import numpy as np
import urllib.parse as urlparse
from urllib.parse import parse_qs
from nltk.stem.wordnet import WordNetLemmatizer
import nltk.data


INDEX_NAME='pubmedx1.6'




### ML training data
def training(request):
	return render(request, 'search/training.html')

def is_conceptid(conceptid, cursor):
	query = "select conceptid from annotation.active_selected_concepts where conceptid =%s "
	res = pg.return_df_from_query(cursor, query, (conceptid,), ["conceptid"])
	if len(res.index) == 1:
		return True
	else:
		return False

def not_in_training_set(condition_id, treatment_id, cursor):
	query = "select condition_id from annotation.training_sentences where condition_id=%s and treatment_id=%s"
	res = pg.return_df_from_query(cursor, query, (condition_id, treatment_id), ['condition_id'])
	if len(res.index) > 0:
		return False
	else:
		return True


def post_training(request):
	conn,cursor = pg.return_postgres_cursor()
	condition_id = request.POST['condition_id']

	treatment_id = request.POST['treatment_id']
	label = None

	if (is_conceptid(condition_id, cursor) and is_conceptid(treatment_id, cursor)):
		if not_in_training_set(condition_id, treatment_id, cursor):
			if request.POST['label'] == '0':
				label = 0
			elif request.POST['label'] == '1':
				label = 1
			elif request.POST['label'] == '2':
				label = 2

			u.treatment_label(condition_id, treatment_id, label, cursor)
			
	cursor.close()
	conn.close()
	return HttpResponseRedirect(reverse('search:training'))



def ml(request):
	return render(request, 'search/ml.html')

def post_ml(request):
	conn, cursor = pg.return_postgres_cursor()
	
	input_sentence = request.POST['sentence']

	term = ann.clean_text(input_sentence)
	all_words = ann.get_all_words_list(term)
	cache = ann.get_cache(all_words, False, cursor)
	
	annotation, sentences = ann.annotate_text_not_parallel(input_sentence, 'unlabelled', cache, cursor, True, True, False)
	annotation = ann.acronym_check(annotation)
	sentence_tuple = ann.get_sentence_annotation(term, annotation)


	res = None
	if request.POST['index']:
		
		condition_ind = int(request.POST['index'])
		condition_id = sentence_tuple[condition_ind][1]
		sentence_df = pd.DataFrame([[term, sentence_tuple]], columns=['sentence', 'sentence_tuples'])
		res = m.analyze_sentence(sentence_df, condition_id, cursor)

	cursor.close()
	conn.close()

	return render(request, 'search/ml.html', {'sentence_tuple' : sentence_tuple, 'res': res, 'input_sentence' : input_sentence})
	# return HttpResponseRedirect(reverse('search:ml'))


### CONCEPT OVERRIDE FUNCTIONS
def concept_override(request):
	return render(request, 'search/concept_override.html')

def get_df_dict(df):
	cols = df.columns.tolist()
	res_arr = []
	for i,t in df.iterrows():
		res_dict = {}
		for j in cols:
			res_dict[j] = t[j]
		res_arr.append(res_dict)
	return res_arr

def post_concept_override(request):
	conn,cursor = pg.return_postgres_cursor()
	payload_dict = {}
	adid = ""
	acid = "" 
	if 'acid' in request.POST:
		acid = request.POST['acid']
		query = """
			select
				t1.adid
				,t1.acid
				,t2.cid
				,t1.term
				,t1.term_lower
				,t1.word
				,t1.word_ord
				,t1.is_acronym
			from annotation2.lemmas t1
			left join annotation2.downstream_root_cid t2
			on t1.acid = t2.acid
			where t1.acid = %s
		"""
		df = pg.return_df_from_query(cursor, query, (acid,), \
	 		["adid", "acid", "cid", "term", "term_lower",'word', "word_ord", "is_acronym"])
		payload_dict['acid'] = get_df_dict(df)
	elif 'adid' in request.POST:
		adid = request.POST['adid']
		query = """
			select
				t1.adid
				,t1.acid
				,t2.cid
				,t1.term
				,t1.term_lower
				,t1.word
				,t1.word_ord
				,t1.is_acronym
			from annotation2.lemmas t1
			left join annotation2.downstream_root_cid t2
			on t1.acid = t2.acid
			where adid = %s
		"""
		df = pg.return_df_from_query(cursor, query, (adid,), \
	 		["adid", "acid", "cid", "term", "term_lower",'word', "word_ord", "is_acronym"])
		payload_dict['adid'] = get_df_dict(df)
	elif 'term' in request.POST:
		term = request.POST['term']
		query = """
			select 
				t1.adid
				,t1.acid
				,t2.cid
				,t1.term
				,t1.term_lower
				,t1.word
				,t1.word_ord
				,t1.is_acronym
			from annotation2.lemmas t1
			left join annotation2.downstream_root_cid t2
			on t1.acid = t2.acid
			where term_lower = lower(%s)
		"""
		df = pg.return_df_from_query(cursor, query, (term,), \
	 		["adid", "acid", "cid", "term", "term_lower",'word', "word_ord", "is_acronym"])
		payload_dict['term'] = get_df_dict(df)
	elif 'acid_relationship' in request.POST:
		acid = request.POST['acid_relationship']
		message = ""
		query = """
			select
				source_acid as item
				,t3.term as item_name
				,destination_acid as parent
				,t2.term as parent_name
			from snomed2.full_relationship_acid t1
			join annotation2.downstream_root_did t2
			on t1.destination_acid = t2.acid
			join annotation2.downstream_root_did t3
			on t1.source_acid = t3.acid
			where source_acid = %s
		"""
		df = pg.return_df_from_query(cursor, query, (acid,), ['item', 'item_name', 'parent', 'parent_name'])
		if len(df.index) == 0:
			message = "No parents found."
		else:
			payload_dict['acid_relationship_parent'] = get_df_dict(df)

		query = """
			select
				source_acid as child
				,t3.term as child_name
				,destination_acid as item
				,t2.term as item_name
			from snomed2.full_relationship_acid t1
			join annotation2.downstream_root_did t2
			on t1.destination_acid = t2.acid
			join annotation2.downstream_root_did t3
			on t1.source_acid = t3.acid
			where destination_acid = %s
		"""
		df = pg.return_df_from_query(cursor, query, (acid,), ['child', 'child_name', 'item', 'item_name'])

		if len(df.index) == 0:
			message += " No children found"
		else:
			payload_dict['acid_relationship_child'] = get_df_dict(df)
		payload_dict['relationship_lookup_message'] = message
	elif 'child_acid' in request.POST: 
		child_acid = request.POST['child_acid']
		parent_acid = request.POST['parent_acid']
		if request.POST['rel_action_type'] == 'add':
			error,message = u2.change_relationship(child_acid, parent_acid, '1', cursor)
		elif request.POST['rel_action_type'] == 'del':
			error,message = u2.change_relationship(child_acid, parent_acid, '0', cursor)
		else:
			error = True
			message = "Action type inappropriately entered"
		payload_dict['change_relationship_error'] = error
		payload_dict['change_relationship_message'] = message
	elif 'new_concept' in request.POST:
		concept_name = request.POST['new_concept']
		error, message, conflict_df = u2.add_new_concept(concept_name, cursor)
		payload_dict['new_concept_message'] = message
		payload_dict['new_concept_success'] = error
		if error:
			payload_dict['new_concept_success'] = False
			payload_dict['new_concept'] = get_df_dict(conflict_df)
		else:
			payload_dict['new_concept_success'] = True
	elif 'remove_acid' in request.POST:
		acid = request.POST['remove_acid']
		error = u2.remove_concept(acid, cursor)
		payload_dict['remove_concept'] = error
	elif 'new_description' in request.POST:
		acid = request.POST['new_description_acid']
		new_description = request.POST['new_description']
		error, message = u2.add_new_description(acid, new_description, cursor)
		payload_dict['add_description_message'] = message
	elif 'remove_adid' in request.POST:
		adid = request.POST['remove_adid']
		error = u2.remove_adid(adid, cursor)
		payload_dict['remove_adid'] = error
	elif 'condition_acid_labelled' in request.POST:
		condition_acid = None
		treatment_acid = None
		if request.POST['condition_acid_labelled'] != '':
			condition_acid = request.POST['condition_acid_labelled']

		if request.POST['treatment_acid_labelled'] != '':
			treatment_acid = request.POST['treatment_acid_labelled']

		relationship = None
		if 'rel_0' in request.POST:
			relationship = 0
		elif 'rel_1' in request.POST:
			relationship = 1
		elif 'rel_2' in request.POST: 
			relationship = 2
		message = u2.add_labelled_treatment(condition_acid, treatment_acid, relationship, cursor)
		payload_dict['labelled_condition_treatment_message'] = message
	elif 'adid_acronym_override' in request.POST:
		adid = request.POST['adid_acronym_override']
		is_acronym = None
		if 'rel_true' in request.POST:
			is_acronym = True
		elif 'rel_false' in request.POST:
			is_acronym = False

		message = u2.acronym_override(adid, is_acronym, cursor)
		payload_dict['acronym_override_message'] = message


	
		


	cursor.close()
	conn.close()

	return render(request, 'search/concept_override.html', {'payload' : payload_dict, 'acid' : acid, 'adid' : adid})


### CONCEPT SEARCH


def concept_search_home_page(request):
	return render(request, 'search/concept_search_home_page.html', 
			{'sr_payload' : None, 'query' : '', 'query_annotation' : None, 'unmatched_terms' : None, 'concepts' : None, 'primary_cids' : None,
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
	
def post_search_text(request):

	conn, cursor = pg.return_postgres_cursor()
	es = es_util.get_es_client()
	pivot_cid = None
	unmatched_terms = ''
	pivot_term = None
	query_type = ''
	primary_cids = None
	query = ''
	filters = {}
	query = None
	
	# If GET request, this is from a link
	if request.method == 'GET': 
		parsed = urlparse.urlparse(request.path)
		parsed = parse_qs(parsed.path)

		# query = parse_qs(parsed.path)['/search/query'][0]

		query = parsed['/search/query'][0]

		if 'query_annotation[]' in parsed:
			primary_cids = parsed['query_annotation[]']
		
		if unmatched_terms in parsed:
			unmatched_terms = parsed['unmatched_terms'][0]
		
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
		unmatched_terms = data['unmatched_terms']
		print(unmatched_terms)

		if 'query_annotation' in data:
			primary_cids = data['query_annotation']

		if 'pivot_cid' in data:
			if data['pivot_cid'] is not None:
				primary_cids.append(data['pivot_cid'])


	sr = dict()
	query_concepts_dict = dict()
	related_dict = {}
	treatment_dict = {}
	condition_dict = {}
	diagnostic_dict = {}
	cause_dict = {}
	calcs_json = {}

	flattened_query = None

	if query_type == 'pivot':
		query_concepts_df = pd.DataFrame(primary_cids, columns=['acid'])

		full_query_concepts_list = ann2.query_expansion(query_concepts_df['acid'], cursor)

		pivot_term = None


		if request.method == 'POST':
			pivot_term = data['pivot_term']
			query = query + ' ' + pivot_term
		else:
			query = query + ' ' + parsed['pivot_term'][0]
		
		params = filters


		flattened_query_concepts_list = get_flattened_query_concept_list(query_concepts_df)
		query_concepts_dict = get_query_arr_dict(full_query_concepts_list)

		es_query = {"from" : 0, \
				 "size" : 100, \
				 "query": get_query(full_query_concepts_list, unmatched_terms, filters['journals'], filters['start_year'], filters['end_year'], \
				 	["title_conceptids^5", "abstract_conceptids.*"], cursor)}

		sr = es.search(index=INDEX_NAME, body=es_query, request_timeout=100000)

		sr_payload = get_sr_payload(sr['hits']['hits'])

		treatment_dict, diagnostic_dict, condition_dict, cause_dict = get_related_conceptids(full_query_concepts_list, primary_cids,
					unmatched_terms, filters, cursor)

		

	elif query_type == 'keyword':

		query = ann2.clean_text(query)
		

		original_query_concepts_list = []
		# if query.upper() != query:
		query = query.lower()

		raw_query = "select acid from annotation2.lemmas where term_lower=%s limit 1"
		query_concepts_df = pg.return_df_from_query(cursor, raw_query, (query,), ["acid"])
		

		if (len(query_concepts_df.index) != 0):
			query_concepts_df['term_start_index'] = 0
			query_concepts_df['term_end_index'] = len(query.split(' '))-1

		elif (len(query_concepts_df.index) == 0):
			
			all_words = ann2.get_all_words_list(query)
			cache = ann2.get_cache(all_words, False)
			
			query_df = return_section_sentences(query, 'query', 0, pd.DataFrame())
			query_concepts_df = ann2.annotate_text_not_parallel(query_df, cache, False, True, False)

		primary_cids = None
		

		if not query_concepts_df['acid'].isnull().all():
			original_query_concepts_list = query_concepts_df['acid'].tolist()
			query_concepts_df = query_concepts_df[query_concepts_df['acid'].notna()].copy()
			unmatched_terms = get_unmatched_terms(query, query_concepts_df)

			full_query_concepts_list = ann2.query_expansion(query_concepts_df['acid'], cursor)
			flattened_query = get_flattened_query_concept_list(full_query_concepts_list)
			query_concept_count = len(query_concepts_df.index)
			query_concepts_dict = get_query_arr_dict(full_query_concepts_list)

			es_query = {"from" : 0, \
						 "size" : 25, \
						 "query": get_query(full_query_concepts_list, unmatched_terms, \
						 	filters['journals'], filters['start_year'], filters['end_year'] \
						 	,["title_conceptids^10", "abstract_conceptids.*^0.5"], cursor)}
			sr = es.search(index=INDEX_NAME, body=es_query, request_timeout=100000)


			treatment_dict, diagnostic_dict, condition_dict, cause_dict = get_related_conceptids(full_query_concepts_list, original_query_concepts_list,
					unmatched_terms, filters, cursor)

			primary_cids = query_concepts_df['acid'].tolist()

			###UPDATE QUERY BELOW FOR FILTERS
		else:
			unmatched_terms = query
			es_query = get_text_query(query)
			sr = es.search(index=INDEX_NAME, body=es_query)

		sr_payload = get_sr_payload(sr['hits']['hits'])
	
	
	calcs_json = get_calcs(query_concepts_df, cursor)
	ip = get_ip_address(request)
	log_query(ip, query, primary_cids, unmatched_terms, filters, treatment_dict, diagnostic_dict, condition_dict, cause_dict, cursor)
	cursor.close()
	conn.close()

	if request.method == 'POST':

		html = render(request, 'search/concept_search_results_page.html', {'sr_payload' : sr_payload, 'query' : query, 'query_annotation' : query_concepts_df['acid'].tolist(), \
				'unmatched_terms' : unmatched_terms, 'concepts' : query_concepts_dict, \
				'journals': filters['journals'], 'start_year' : filters['start_year'], 'end_year' : filters['end_year'], \
				'treatment' : treatment_dict, 'diagnostic' : diagnostic_dict, 'cause' : cause_dict, 'condition' : condition_dict, \
				'calcs' : calcs_json})
		
		return html
	elif request.method == 'GET':
	
		return render(request, 'search/concept_search_home_page.html', {'sr_payload' : sr_payload, 'query' : query, 'query_annotation' : query_concepts_df['acid'].tolist(), \
				'unmatched_terms' : unmatched_terms, 'concepts' : query_concepts_dict, \
				'journals': filters['journals'], 'start_year' : filters['start_year'], 'end_year' : filters['end_year'], \
				'treatment' : treatment_dict, 'diagnostic' : diagnostic_dict, 'cause' : cause_dict, 'condition' : condition_dict, \
				'calcs' : calcs_json})

def get_ip_address(request):
	ip = ''
	x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
	if x_forwarded_for:
		ip = x_forwarded_for.split(',')[0]
	else:
		ip = request.META.get('REMOTE_ADDR')
	return ip

def get_calcs(query_concepts_df, cursor):
	concepts = query_concepts_df[query_concepts_df['acid'].notna()].copy()
	concepts = concepts['acid'].tolist()
	if len(concepts) > 0:
		query = "select distinct on (title, t1.description, url) title, t1.description, url from annotation2.mdc_final t1 where acid in %s"
		calcs = pg.return_df_from_query(cursor, query, (tuple(concepts),), ['title', 'desc', 'url'])

		calc_json = []
		for ind,item in calcs.iterrows():
			calc_json.append({'title' : item['title'], 'desc' : item['desc'], 'url' : item['url']})

		return calc_json
	else:
		return None

def rollups(cids_df, cursor):
	if len(cids_df.index) > 0:
		params = (tuple(cids_df['acid']), tuple(cids_df['acid']))
		query = """ 
			select 
				child_acid
				,parent_acid 
			from snomed2.transitive_closure_acid 
			where child_acid in %s and parent_acid in %s 
			"""

		#and parent_acid not in (select parent_acid from snomed2.transitive_closure_acid group by parent_acid having count(*) > 1000)

		parents_df = pg.return_df_from_query(cursor, query, params, ["child_acid", "parent_acid"])

		parents_df = parents_df[parents_df['parent_acid'].isin(parents_df['child_acid'].tolist()) == False].copy()


		orphan_df = cids_df[(cids_df['acid'].isin(parents_df['child_acid'].tolist()) == False) 
			& (cids_df['acid'].isin(parents_df['parent_acid'].tolist()) == False)]
		orphan_df = orphan_df.sort_values(by=['count'], ascending=False)

		json = []
		for ind,parent in parents_df.iterrows():
			added = False
			for item in json:
				if parent['parent_acid'] in item.keys():
					cnt = int(cids_df[cids_df['acid'] == parent['child_acid']]['count'].values[0])
					item[parent['parent_acid']]['children'].append(cids_df[cids_df['acid']==parent['child_acid']]['term'].values[0])
					item[parent['parent_acid']]['count'] = cnt + int(item[parent['parent_acid']]['count'])
					added = True
			if not added:
				cnt = int(cids_df[cids_df['acid'] == parent['parent_acid']]['count'].values[0])
				cnt = cnt + int(cids_df[cids_df['acid'] == parent['child_acid']]['count'].values[0])
				r = {parent['parent_acid'] : {'name' : cids_df[cids_df['acid'] == parent['parent_acid']]['term'].values[0],\
					 'count' : cnt, 'children' : [cids_df[cids_df['acid']==parent['child_acid']]['term'].values[0]]}}
				json.append(r)

		for ind, item in orphan_df.iterrows():
			r = {item['acid'] : {'name' : item['term'], 'count' : item['count'], 'children' : []}}
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

def log_query (ip_address, query, primary_cids, unmatched_terms, filters, treatment_dict, diagnostic_dict, condition_dict, cause_dict, cursor):
	query = """
		insert into search.query_logs
		VALUES
		(%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s)
	"""
	cursor.execute(query, (ip_address,query,json.dumps(primary_cids), \
		unmatched_terms,filters['start_year'], filters['end_year'], json.dumps(filters['journals']), \
		json.dumps(condition_dict), json.dumps(treatment_dict), json.dumps(diagnostic_dict), json.dumps(cause_dict)))

	cursor.connection.commit()



### Utility functions

def get_sr_payload(sr):
	sr_list = []

	for index,hit in enumerate(sr):
		hit_dict = {}
		sr_src = hit['_source']
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


def get_unmatched_terms(query, query_concepts_df):
	unmatched_terms = ''
	for index,word in enumerate(query.split()):
		if not query_concepts_df.empty and len((query_concepts_df[(query_concepts_df['term_end_index'] >= index) & (query_concepts_df['term_start_index'] <= index)]).index) > 0:
			continue
		else:
			unmatched_terms += word + " "
	return unmatched_terms




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
			{"term": {'article_title': "rats"}}
			]
	return filt

def get_concept_string(conceptid_series):
	result_string = ""
	for item in conceptid_series:
		result_string += item + " "
	
	return result_string.strip()

def get_query(full_conceptid_list, unmatched_terms, journals, start_year, end_year, fields_arr, cursor):
	es_query = {}
	if unmatched_terms == '':
		es_query["bool"] = { \
							"must_not": get_article_type_filters(), \
							"must": \
								[{"query_string": {"fields" : fields_arr, \
								 "query" : get_concept_query_string(full_conceptid_list, cursor)}}]}
	else:

		es_query["bool"] = { \
						"must_not": get_article_type_filters(), \
						"must": \
							[{"query_string": {"fields" : fields_arr, \
							 "query" : get_concept_query_string(full_conceptid_list, cursor)}}, {"query_string": {"fields" : ["article_title", "article_abstract.*"], \
							"query" : unmatched_terms}}]}
						# "should": \
						# 	[{"query_string": {"fields" : fields_arr, \
						# 	"query" : unmatched_terms}}]}


	if (len(journals) > 0) or start_year or end_year:
		d = []

		if len(journals) > 0:

			for i in journals:
				d.append({"term" : {'journal_iso_abbrev' : i}})

		if start_year and end_year:
			d.append({"range" : {"journal_pub_year" : {"gte" : start_year, "lte" : end_year}}})
		elif start_year:
			d.append({"range" : {"journal_pub_year" : {"gte" : start_year}}})
		elif end_year:
			d.append({"range" : {"journal_pub_year" : {"lte" : end_year}}})

		es_query["bool"]["filter"] = d



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
			 "size" : 500, \
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
				flattened_concept_df = flattened_concept_df.append(pd.DataFrame([[concept]], columns=['acid']))
		else:
			flattened_concept_df = flattened_concept_df.append(pd.DataFrame([[item]], columns=['acid']))

	flattened_concept_df = ann2.add_names(flattened_concept_df)

	dict_arr = []
	for index,row in flattened_concept_df.iterrows():
		concept_dict = {'acid' : row['acid'], 'term' : row['term']}
		dict_arr.append(concept_dict)
	if len(dict_arr) == 0:
		return None
	else:
		return dict_arr

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
	concept_type_query_string = "select root_cid as conceptid, rel_type as concept_type from annotation2.concept_types where root_cid in %s"

	query_concept_type_df = pg.return_df_from_query(cursor, concept_type_query_string, \
		(tuple(flattened_concept_list),), ["conceptid", "concept_type"])

	return query_concept_type_df


# Conceptid_df is the title_match
def get_query_concept_types_df_3(conceptid_df, query_concept_list, cursor, concept_type):

	dist_concept_list = list(set(conceptid_df['acid'].tolist()))

	if concept_type == 'treatment' and len(dist_concept_list) > 0:
		query = """

			select 
				treatment_acid 
			from ml2.treatment_recs_final 
			where condition_acid in %s and treatment_acid not in 
				(select treatment_acid from ml2.labelled_treatments where label=2 and treatment_acid is not NULL)
		"""
		tx_df = pg.return_df_from_query(cursor, query, (tuple(query_concept_list),), ["acid"])

		conceptid_df = pd.merge(conceptid_df, tx_df, how='inner', on=['acid'])

		return conceptid_df

	elif len(dist_concept_list) > 0:

		concept_type_query_string = """
			select 
				root_acid as acid
				,rel_type as concept_type
			from annotation2.base_concept_types
			where rel_type = %s
		"""

		query_concept_type_df = pg.return_df_from_query(cursor, concept_type_query_string, (concept_type,), ["acid", "concept_type"])
		conceptid_df = pd.merge(conceptid_df, query_concept_type_df, how='inner', on=['acid'])

		return conceptid_df
	else:
		concept_type_query_string = """
			select root_cid as acid
			from annotation2.concept_types
			where rel_type=%s

		"""
		query_concept_type_df = pg.return_df_from_query(cursor, concept_type_query_string, (concept_type,), ["acid"])
		conceptid_df = pd.merge(conceptid_df, query_concept_type_df, how='right', on=['acid'])
		return conceptid_df


def get_related_conceptids(query_concept_list, original_query_concepts_list, unmatched_terms, filters, cursor):
	result_dict = dict()
	es = es_util.get_es_client()

	es_query = get_query(query_concept_list, unmatched_terms, \
						 	filters['journals'], filters['start_year'], filters['end_year']\
						 	,["title_conceptids^10", "abstract_conceptids.*^0.5"], cursor)

	scroller = es_util.ElasticScroll(es, es_query)

	title_match_cids_df = pd.DataFrame()
	# while scroller.has_next:
	counter = 0
	while counter < 2:
		article_list = scroller.next()
		if article_list is not None: 
			title_match_cids_df = title_match_cids_df.append(get_title_cids(article_list), sort=False)
		else:
			break
		counter += 1

	title_match_cids_df = title_match_cids_df[title_match_cids_df['acid'].isin(original_query_concepts_list) == False].copy()
	filter_concepts = ['11220']
	title_match_cids_df = title_match_cids_df[title_match_cids_df['acid'].isin(filter_concepts) == False].copy()

	sub_dict = dict()
	sub_dict['treatment'] = []
	sub_dict['diagnostic'] = []
	sub_dict['condition'] = []
	sub_dict['cause'] = []

	if len(title_match_cids_df) > 0:

		flattened_query_concepts = get_flattened_query_concept_list(query_concept_list)

		agg_tx = get_query_concept_types_df_3(title_match_cids_df, flattened_query_concepts, cursor, 'treatment')

		if not agg_tx.empty:
			agg_tx = agg_tx.drop_duplicates(subset=['acid', 'pmid'])
			agg_tx['count'] = 1
			agg_tx = agg_tx.groupby(['acid'], as_index=False)['count'].sum()

			agg_tx = ann2.add_names(agg_tx)
			sub_dict['treatment'] = rollups(agg_tx, cursor)


		agg_diagnostic = get_query_concept_types_df_3(title_match_cids_df, flattened_query_concepts, cursor, 'diagnostic')

		if len(agg_diagnostic) > 0:
			agg_diagnostic = agg_diagnostic.drop_duplicates(subset=['acid', 'pmid'])
			agg_diagnostic['count'] = 1
			agg_diagnostic = agg_diagnostic.groupby(['acid'],  as_index=False)['count'].sum()
			
			agg_diagnostic = ann2.add_names(agg_diagnostic)
			sub_dict['diagnostic'] = rollups(agg_diagnostic, cursor)
			

		agg_cause = get_query_concept_types_df_3(title_match_cids_df, flattened_query_concepts, cursor, 'cause')

		if len(agg_cause) > 0:
			agg_cause = agg_cause.drop_duplicates(subset=['acid', 'pmid'])
			agg_cause['count'] = 1
			agg_cause = agg_cause.groupby(['acid'],  as_index=False)['count'].sum()
			agg_cause = ann2.add_names(agg_cause)
			sub_dict['cause'] = rollups(agg_cause, cursor)


		agg_condition = get_query_concept_types_df_3(title_match_cids_df, flattened_query_concepts, cursor, 'condition')

		if len(agg_condition) > 0:
			agg_condition = agg_condition.drop_duplicates(subset=['acid', 'pmid'])
			agg_condition['count'] = 1
			agg_condition = agg_condition.groupby(['acid'],  as_index=False)['count'].sum()
			agg_condition = ann2.add_names(agg_condition)
			agg_condition = agg_condition.sort_values(['count'], ascending=False)
			u2.pprint(agg_condition)
			sub_dict['condition'] = rollups(agg_condition, cursor)

	return sub_dict['treatment'], sub_dict['diagnostic'], sub_dict['condition'], sub_dict['cause']

# this function isn't working - see schizophrenia treatment
def de_dupe_synonyms(df, cursor):

	synonyms = ann.get_concept_synonyms_df_from_series(df['conceptid'], cursor)

	for ind,t in df.iterrows():
		cnt = df[df['conceptid'] == t['conceptid']]
		ref = synonyms[synonyms['reference_conceptid'] == t['conceptid']]

		if len(ref) > 0:
			new_conceptid = ref.iloc[0]['synonym_conceptid']
			if len(df[df['conceptid'] == new_conceptid].index):
				
				df.loc[ind, 'conceptid'] = new_conceptid

	df = df.groupby(['conceptid'], as_index=False)['count'].sum()

	return df 		

def de_dupe_synonyms_2(df, cursor):
	if len(df) > 0:

		synonym_query = """
			select 
			distinct reference_conceptid, synonym_conceptid
			from (
				select
				t3.reference_conceptid
				,case when t3.reference_rank < t3.synonym_rank then t3.reference_conceptid
				when t3.reference_rank >= t3.synonym_rank then t3.synonym_conceptid 
				end as synonym_conceptid
			from (
				select t1.reference_conceptid, t1.reference_term, min(t1.synonym_rank) as mini
				from annotation.concept_terms_synonyms t1
				where t1.reference_conceptid in %s
				group by t1.reference_conceptid, t1.reference_term
			) t2
			join annotation.concept_terms_synonyms t3
			on t2.reference_conceptid = t3.reference_conceptid and t2.mini = t3.synonym_rank
		) t4
		"""
		synonyms = pg.return_df_from_query(cursor, synonym_query, (tuple(df['conceptid'].tolist()),), ['reference_conceptid', 'synonym_conceptid'])

		results_df = pd.DataFrame()		
		for ind,t in df.iterrows():

			synonym_cid = synonyms[synonyms['reference_conceptid'] == t['conceptid']]['synonym_conceptid'].values

			if len(synonym_cid) > 0:
				synonym_cid = synonym_cid[0]

				if t['conceptid'] not in synonym_cid:
					results_df = results_df.append(pd.DataFrame([[synonym_cid, t['pmid']]], columns=['conceptid', 'pmid']))
				else:
					results_df = results_df.append(t)
			else:
				results_df = results_df.append(t)


		return results_df 
	else:
		return None

def get_conceptids_from_sr(sr):
	conceptid_list = []

	for hit in sr['hits']['hits']:
		if hit['_source']['title_conceptids'] is not None:
			conceptid_list.extend(hit['_source']['title_conceptids'])
		if hit['_source']['abstract_conceptids'] is not None:
			for key1 in hit['_source']['abstract_conceptids']:
				if hit['_source']['abstract_conceptids'][key1] is not None:
					conceptid_list.extend(hit['_source']['abstract_conceptids'][key1])
	return conceptid_list

def get_title_cids(sr):

	conceptid_df = pd.DataFrame(columns=['acid', 'pmid'])
	for hit in sr['hits']['hits']:
		if hit['_source']['title_conceptids'] is not None:
			pmid = hit['_source']['pmid']
			cid_list = list(set(hit['_source']['title_conceptids']))
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
