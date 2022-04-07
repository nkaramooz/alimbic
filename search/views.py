from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from django.utils.safestring import mark_safe
import snomed_annotator2 as ann2
import utilities.pglib as pg
from nltk.stem.wordnet import WordNetLemmatizer
import utilities.utils2 as u
import pandas as pd
import utilities.es_utilities as es_util
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
lmtzr.lemmatize("cough")

### ML training data

def labelling(request):
	conn,cursor = pg.return_postgres_cursor()
	query = """
		select
			condition_acid
			,condition
			,treatment_acid
			,treatment
			,t1avg
			,t2avg
		from ml2.rapid_labelling
		where ver=0
		limit 1
	"""
	entry = pg.return_df_from_query(cursor, query, None,
		['condition_acid', 'condition', 'treatment_acid', 'treatment',
		't1avg', 't2avg'])

	cursor.close()
	conn.close()
	return render(request, 'search/labelling.html', 
		{'condition_acid': entry['condition_acid'][0], 'condition': entry['condition'][0],
		'treatment_acid': entry['treatment_acid'][0], 'treatment': entry['treatment'][0],
		't1avg' : entry['t1avg'][0], 't2avg': entry['t2avg'][0]})

def post_labelling(request):
	conn,cursor = pg.return_postgres_cursor()
	payload_dict = {}
	condition_acid = request.POST['condition_acid']
	treatment_acid = request.POST['treatment_acid']

	if condition_acid != None and treatment_acid != None:
		message = None
		label = None
		if 'rel_0' in request.POST:
			label = 0
		elif 'rel_1' in request.POST:
			label = 1
		elif 'rel_2' in request.POST: 
			label = 2
		if label == 2:
			message = "Skipped"
		else:
			message = u.add_labelled_treatment(condition_acid, treatment_acid, label, cursor)
		if label != None:
			query = """
					UPDATE ml2.rapid_labelling
					set ver=1
					where condition_acid=%s 
					and treatment_acid=%s
				"""
			cursor.execute(query, (condition_acid,treatment_acid))
			cursor.connection.commit()
	cursor.close()
	conn.close()
	# return render(request, 'search/labelling.html', {'message' : message})
	return HttpResponseRedirect(reverse('search:labelling'))


def training(request):
	conn,cursor = pg.return_postgres_cursor()
	query = """
		select
			entry_id
			,sentence_id
			,sentence_tuples
			,label
			,ver
		from ml2.manual_spacy_labels
		where ver = 0
		order by random()
		limit 1
	"""
	entry = pg.return_df_from_query(cursor, query, None, \
		['entry_id', 'sentence_id', 'sentence_tuples', 'label', 'ver'])

	sentence = ''
	for item in entry['sentence_tuples'][0]:
		sentence += item[0] + ' '

	conn.close()
	cursor.close()

	return render(request, 'search/training.html', \
		{'entry_id' : entry['entry_id'][0], 'sentence_id' : entry['sentence_id'][0], \
		 'sentence': sentence, 'sentence_tuples' : entry['sentence_tuples'][0], 'label' : entry['label'], 'ver' : entry['ver'][0]})



def post_training(request):
	conn,cursor = pg.return_postgres_cursor()
	
	
	sentence = request.POST['sentence']

	labels = []
	if request.POST['start1'] != '' and request.POST['end1'] != '' \
		and 'label1' in request.POST:
		labels.append((int(request.POST['start1']), int(request.POST['end1']), request.POST['label1']))

	if request.POST['start2'] != '' and request.POST['end2'] != '' \
		and 'label2' in request.POST:
		labels.append((int(request.POST['start2']), int(request.POST['end2']), request.POST['label2']))

	if request.POST['start3'] != '' and request.POST['end3'] != '' \
		and 'label3' in request.POST:
		labels.append((int(request.POST['start3']), int(request.POST['end3']), request.POST['label3']))

	if request.POST['start4'] != '' and request.POST['end4'] != '' \
		and 'label4' in request.POST:
		labels.append((int(request.POST['start4']), int(request.POST['end4']), request.POST['label4']))

	if request.POST['start5'] != '' and request.POST['end5'] != '' \
		and 'label5' in request.POST:
		labels.append((int(request.POST['start5']), int(request.POST['end5']), request.POST['label5']))

	if request.POST['start6'] != '' and request.POST['end6'] != '' \
		and 'label6' in request.POST:
		labels.append((int(request.POST['start6']), int(request.POST['end6']), request.POST['label6']))

	if request.POST['start7'] != '' and request.POST['end7'] != '' \
		and 'label7' in request.POST:
		labels.append((int(request.POST['start7']), int(request.POST['end7']), request.POST['label7']))

	if request.POST['start8'] != '' and request.POST['end8'] != '' \
		and 'label8' in request.POST:
		labels.append((int(request.POST['start8']), int(request.POST['end8']), request.POST['label8']))

	if request.POST['start9'] != '' and request.POST['end9'] != '' \
		and 'label9' in request.POST:
		labels.append((int(request.POST['start9']), int(request.POST['end9']), request.POST['label9']))

	if request.POST['start10'] != '' and request.POST['end10'] != '' \
		and 'label10' in request.POST:
		labels.append((int(request.POST['start10']), int(request.POST['end10']), request.POST['label10']))
	
	if len(labels) > 0:
		entry_id = request.POST['entry_id']
		query = """
			update ml2.manual_spacy_labels
				set 
					spacy_label = %s
					,ver = 1
				where entry_id = %s
		"""
		labels = (sentence, {"entities" : labels})
		cursor.execute(query, (json.dumps(labels), entry_id))
		cursor.connection.commit()

	cursor.close()
	conn.close()
	return HttpResponseRedirect(reverse('search:training'))




def is_conceptid(conceptid, cursor):
	query = "select conceptid from annotation2.active_selected_concepts where conceptid =%s "
	res = pg.return_df_from_query(cursor, query, (conceptid,), ["conceptid"])
	if len(res.index) == 1:
		return True
	else:
		return False


def ml(request):
	return render(request, 'search/ml.html')

def post_ml(request):
	conn, cursor = pg.return_postgres_cursor()
	
	input_sentence = request.POST['sentence']

	term = ann.clean_text(input_sentence)
	all_words = ann.get_all_words_list(term)
	cache = ann.get_cache(all_words, False, cursor, lmtzr)
	
	annotation, sentences = ann.annotate_text_not_parallel(input_sentence, 'unlabelled', cache, cursor, True, True, False, lmtzr)
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
		payload_dict['acid'] = df.to_dict('records')
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
		payload_dict['adid'] = df.to_dict('records')
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
		payload_dict['term'] = df.to_dict('records')
	elif 'acid_relationship' in request.POST:
		acid = request.POST['acid_relationship']
		message = ""
		# query = """
		# 	select
		# 		child_acid as item
		# 		,t3.term as item_name
		# 		,parent_acid as parent
		# 		,t2.term as parent_name
		# 	from snomed2.transitive_closure_acid t1
		# 	join (select distinct on (acid) acid, term from annotation2.downstream_root_did) t2
		# 	on t1.parent_acid = t2.acid
		# 	join (select distinct on (acid) acid, term from annotation2.downstream_root_did) t3
		# 	on t1.child_acid = t3.acid
		# 	where child_acid = %s
		# """
		query = """
			select
				source_acid as item
				,t3.term as item_name
				,destination_acid as parent
				,t2.term as parent_name
			from snomed2.full_relationship_acid t1
			join (select distinct on (acid) acid, term from annotation2.downstream_root_did) t2
				on t1.destination_acid = t2.acid
			join (select distinct on (acid) acid, term from annotation2.downstream_root_did) t3
				on t1.source_acid = t3.acid
			where source_acid = %s and typeid='116680003' and active='1'
		"""
		df = pg.return_df_from_query(cursor, query, (acid,), ['item', 'item_name', 'parent', 'parent_name'])
		if len(df.index) == 0:
			message = "No parents found."
		else:
			payload_dict['acid_relationship_parent'] = df.to_dict('records')

		# query = """
		# 	select
		# 		child_acid as child
		# 		,t3.term as child_name
		# 		,parent_acid as item
		# 		,t2.term as item_name
		# 	from snomed2.transitive_closure_acid t1
		# 	join (select distinct on (acid) acid, term from annotation2.downstream_root_did) t2
		# 	on t1.parent_acid = t2.acid
		# 	join (select distinct on (acid) acid, term from annotation2.downstream_root_did) t3
		# 	on t1.child_acid = t3.acid
		# 	where parent_acid = %s
		# """
		query = """
			select
				source_acid as child
				,t2.term as child_name
				,destination_acid as item
				,t3.term as item_name
			from snomed2.full_relationship_acid t1
			join (select distinct on (acid) acid, term from annotation2.downstream_root_did) t2
				on t1.source_acid = t2.acid
			join (select distinct on (acid) acid, term from annotation2.downstream_root_did) t3
				on t1.destination_acid = t3.acid
			where destination_acid = %s and typeid='116680003' and active='1'

		"""
		df = pg.return_df_from_query(cursor, query, (acid,), ['child', 'child_name', 'item', 'item_name'])

		if len(df.index) == 0:
			message += " No children found"
		else:
			payload_dict['acid_relationship_child'] = df.to_dict('records')
		payload_dict['relationship_lookup_message'] = message
	elif 'child_acid' in request.POST: 
		child_acid = request.POST['child_acid']
		parent_acid = request.POST['parent_acid']
		if request.POST['rel_action_type'] == 'add':
			error,message = u.change_relationship(child_acid, parent_acid, '1', cursor)
		elif request.POST['rel_action_type'] == 'del':
			error,message = u.change_relationship(child_acid, parent_acid, '0', cursor)
		else:
			error = True
			message = "Action type inappropriately entered"
		payload_dict['change_relationship_error'] = error
		payload_dict['change_relationship_message'] = message
	elif 'new_concept' in request.POST:
		concept_name = request.POST['new_concept']
		error, message, conflict_df = u.add_new_concept(concept_name, cursor)
		payload_dict['new_concept_message'] = message
		payload_dict['new_concept_success'] = error
		if error:
			payload_dict['new_concept_success'] = False
			payload_dict['new_concept'] = conflict_df.to_dict('records')
		else:
			payload_dict['new_concept_success'] = True
	elif 'remove_acid' in request.POST:
		acid = request.POST['remove_acid']
		error = u.remove_concept(acid, cursor)
		payload_dict['remove_concept'] = error
	elif 'new_description' in request.POST:
		acid = request.POST['new_description_acid']
		new_description = request.POST['new_description']
		error, message = u.add_new_description(acid, new_description, cursor)
		payload_dict['add_description_message'] = message
	elif 'remove_adid' in request.POST:
		adid = request.POST['remove_adid']
		error = u.remove_adid(adid, cursor)
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
		message = u.add_labelled_treatment(condition_acid, treatment_acid, relationship, cursor)
		payload_dict['labelled_condition_treatment_message'] = message
	elif 'adid_acronym_override' in request.POST:
		adid = request.POST['adid_acronym_override']
		is_acronym = None
		if 'rel_true' in request.POST:
			is_acronym = True
		elif 'rel_false' in request.POST:
			is_acronym = False

		message = u.acronym_override(adid, is_acronym, cursor)
		payload_dict['acronym_override_message'] = message

	elif 'acid_change_concept_type' in request.POST:
		acid = request.POST['acid_change_concept_type']
		rel_type = None
		state = None
		if 'rel_condition' in request.POST:
			rel_type = 'condition'
		elif 'rel_symptom' in request.POST:
			rel_type = 'symptom'
		elif 'rel_cause' in request.POST:
			rel_type = 'cause'
		elif 'rel_treatment' in request.POST:
			rel_type = 'treatment'
		elif 'rel_diagnostic' in request.POST:
			rel_type = 'diagnostic'
		elif 'rel_statistic' in request.POST:
			rel_type = 'statistic'
		elif 'rel_outcome' in request.POST:
			rel_type = 'outcome'

		if 'rel_activate' in request.POST:
			state = 1
		elif 'rel_inactivate' in request.POST:
			state = 0

		if rel_type is not None and state is not None:
			message = u.change_concept_type(acid, rel_type, state, cursor)
			payload_dict['change_concept_type_message'] = message
		else:
			payload_dict['change_concept_type_message'] = "Error status or rel_type is null"
		


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
	spellcheck_threshold = 100
	# If GET request, this is from a link
	if request.method == 'GET': 
		parsed = urlparse.urlparse(request.path)
		parsed = parse_qs(parsed.path)

		query = parsed['/search/query'][0]

		if 'query_annotation[]' in parsed:
			primary_cids = parsed['query_annotation[]']

		if 'expanded_query_acids[]' in parsed:
			expanded_query_acids = parsed['expanded_query_acids[]']

		if 'pivot_complete_acid[]' in parsed:
			pivot_complete_acid = parsed['pivot_complete_acid[]']

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
		if 'unmatched_terms' in data:
			unmatched_terms = data['unmatched_terms']
		else: 
			unmatched_terms = ''

		if 'query_annotation' in data:
			primary_cids = data['query_annotation']

		if 'expanded_query_acids' in data:
			expanded_query_acids = data['expanded_query_acids']

		if 'pivot_complete_acid' in data:
			pivot_complete_acid = data['pivot_complete_acid']

		if 'pivot_cid' in data:
			if data['pivot_cid'] is not None:
				primary_cids.append(data['pivot_cid'])


	sr = dict()
	related_dict = {}
	treatment_dict = {}
	condition_dict = {}
	diagnostic_dict = {}
	cause_dict = {}
	calcs_json = {}

	flattened_query = None

	if query_type == 'pivot':
		query_concepts_df = pd.DataFrame(primary_cids, columns=['acid'])
		query_concepts_types = get_query_concept_types_df(query_concepts_df['acid'].tolist(), cursor)
		query_types_list = query_concepts_types['concept_type'].tolist()
		expanded_query_acids.extend(pivot_complete_acid)
		full_query_concepts_list = ann2.query_expansion(primary_cids, expanded_query_acids, cursor)

		pivot_term = None

		if request.method == 'POST':
			pivot_term = data['pivot_term']
			query = query + ' ' + pivot_term
		else:
			query = query + ' ' + parsed['pivot_term'][0]
		
		params = filters


		flattened_query_concepts_list = get_flattened_query_concept_list(full_query_concepts_list)

		es_query = {"from" : 0, \
				 "size" : 50, \
				 "query": get_query(full_query_concepts_list, unmatched_terms, query_types_list, filters['journals'], filters['start_year'], filters['end_year'], \
				 	["title_cids^10", "abstract_conceptids.*"], cursor)}

		sr = es.search(index=INDEX_NAME, body=es_query, request_timeout=100000)

		treatment_dict, diagnostic_dict, condition_dict, cause_dict, expanded_query_acids = get_related_conceptids(full_query_concepts_list, primary_cids,\
			flattened_query_concepts_list, query_types_list, unmatched_terms, filters, cursor)
		
		sr_payload = get_sr_payload(sr['hits']['hits'], expanded_query_acids, unmatched_terms, cursor)

		
	elif query_type == 'keyword':
		original_query_concepts_list = []
		raw_query = "select acid from annotation2.lemmas where term_lower=%s limit 1"
		query_concepts_df = pg.return_df_from_query(cursor, raw_query, (query.lower(),), ["acid"])

		if (len(query_concepts_df.index) != 0):
			query_concepts_df['term_start_index'] = 0
			query_concepts_df['term_end_index'] = len(query.split(' '))-1


		elif (len(query_concepts_df.index) == 0):
			query = ann2.clean_text(query)

			all_words = ann2.get_all_words_list(query, lmtzr)
			cache = ann2.get_cache(all_words_list=all_words, case_sensitive=False, \
			check_pos=False, spellcheck_threshold=spellcheck_threshold, lmtzr=lmtzr)

			query_df = return_section_sentences(query, 'query', 0, pd.DataFrame())
			query_concepts_df = ann2.annotate_text_not_parallel(sentences_df=query_df, cache=cache, \
			case_sensitive=False, check_pos=False, bool_acr_check=False,\
			spellcheck_threshold=spellcheck_threshold, \
			write_sentences=False, lmtzr=lmtzr)

		primary_cids = None

		if not query_concepts_df['acid'].isnull().all():
			query_concepts_df = query_concepts_df[query_concepts_df['acid'].notna()].drop_duplicates(subset=['acid']).copy()
			original_query_concepts_list = set(query_concepts_df['acid'].tolist())
			query_types_list = get_query_concept_types_df(original_query_concepts_list, cursor)['concept_type'].tolist()

			unmatched_terms = get_unmatched_terms(query, query_concepts_df)

			full_query_concepts_list = ann2.query_expansion(original_query_concepts_list, None, cursor)

			flattened_query = get_flattened_query_concept_list(full_query_concepts_list)

			query_concept_count = len(query_concepts_df.index)

			es_query = {"from" : 0, \
						 "size" : 50, \
						 "query": get_query(full_query_concepts_list, unmatched_terms, query_types_list \
						 	,filters['journals'], filters['start_year'], filters['end_year'] \
						 	,["title_cids^10", "abstract_conceptids.*^1"], cursor)}

			sr = es.search(index=INDEX_NAME, body=es_query, request_timeout=100000)

			history_query = """
				select 
					expanded_query_acids
					,treatment_json as treatment_dict
					,diagnostic_json as diagnostic_dict
					,condition_json as condition_dict
					,cause_json as cause_dict
				from search.query_logs
				where query=%s and querytime::date = now()::date
				limit 1
			"""

			query_logs_df = pg.return_df_from_query(cursor, history_query, (query,), \
				['expanded_query_acids', 'treatment_dict', 'diagnostic_dict', 'condition_dict', 'cause_dict'])
			
			query_logs_df = pd.DataFrame()

			if len(query_logs_df.index) > 0:
				treatment_dict = query_logs_df['treatment_dict'][0]
				diagnostic_dict = query_logs_df['diagnostic_dict'][0]
				condition_dict = query_logs_df['condition_dict'][0]
				cause_dict = query_logs_df['cause_dict'][0]
				expanded_query_acids = query_logs_df['expanded_query_acids'][0]
			else:			
				treatment_dict, diagnostic_dict, condition_dict, cause_dict, expanded_query_acids = get_related_conceptids(full_query_concepts_list, \
					original_query_concepts_list, flattened_query, query_types_list, unmatched_terms, filters, cursor)
			primary_cids = query_concepts_df['acid'].tolist()

			###UPDATE QUERY BELOW FOR FILTERS
		else:
			unmatched_terms = query
			es_query = get_text_query(query)
			sr = es.search(index=INDEX_NAME, body=es_query)

		i = u.Timer("get_payload; bold")
		sr_payload = get_sr_payload(sr['hits']['hits'], expanded_query_acids, unmatched_terms, cursor)
		i.stop()

	calcs_json = get_calcs(query_concepts_df, cursor)
	ip = get_ip_address(request)
	log_query(ip, query, primary_cids, expanded_query_acids, unmatched_terms, filters, treatment_dict, diagnostic_dict, condition_dict, cause_dict, cursor)
	cursor.close()
	conn.close()

	if request.method == 'POST':

		html = render(request, 'search/concept_search_results_page.html', {'sr_payload' : sr_payload, 
				'query' : query, 'query_annotation' : query_concepts_df['acid'].tolist(),
				'expanded_query_acids' : expanded_query_acids,
				'unmatched_terms' : unmatched_terms, \
				'journals': filters['journals'], 'start_year' : filters['start_year'], 'end_year' : filters['end_year'], \
				'treatment' : treatment_dict, 'diagnostic' : diagnostic_dict, 'cause' : cause_dict, 'condition' : condition_dict, \
				'calcs' : calcs_json})

		return html
	elif request.method == 'GET':

		return render(request, 'search/concept_search_home_page.html', {'sr_payload' : sr_payload, 'query' : query, 
				'query_annotation' : query_concepts_df['acid'].tolist(),
				'expanded_query_acids' : expanded_query_acids,
				'unmatched_terms' : unmatched_terms, 'journals': filters['journals'], 
				'start_year' : filters['start_year'], 'end_year' : filters['end_year'],
				'treatment' : treatment_dict, 'diagnostic' : diagnostic_dict, 'cause' : cause_dict, 'condition' : condition_dict,
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
			(select child_acid from snomed2.transitive_closure_acid where child_acid in %s and parent_acid in %s)
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
		print(distinct_parents)
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

def log_query (ip_address, query, primary_cids, expanded_query_acids, unmatched_terms, filters, treatment_dict, diagnostic_dict, condition_dict, cause_dict, cursor):
	insert_query = """
		insert into search.query_logs
		VALUES(%s, %s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, now())
	"""
	cursor.execute(insert_query, (ip_address, query,json.dumps(primary_cids), \
		json.dumps(expanded_query_acids), unmatched_terms,filters['start_year'], filters['end_year'], json.dumps(filters['journals']), \
		json.dumps(condition_dict), json.dumps(treatment_dict), json.dumps(diagnostic_dict), json.dumps(cause_dict)))

	cursor.connection.commit()



### Utility functions

def get_sr_payload(sr, expanded_query_acids, unmatched_terms, cursor):
	sr_list = []

	terms = []
	if len(expanded_query_acids) > 0:
		query = """
			select term from annotation2.used_descriptions where acid in %s order by length(term) desc
		"""
		terms = pg.return_df_from_query(cursor, query, (tuple(expanded_query_acids),), ["term"])["term"].tolist()

	if unmatched_terms != '':
		terms.append(unmatched_terms)

	for index,hit in enumerate(sr):
		hit_dict = {}
		sr_src = hit['_source']
		for term in terms:
			term_search = r"(\b" + term + r"\b)(?!(.(?!<b))*</b>)"

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
			{"term": {'article_title': "dog"}},
			{"term": {'article_title': "dogs"}},
			{"term": {'article_title': "cat"}},
			{"term": {'article_title': "cats"}},
			{"term": {'article_title': "rabbit"}},
			{"term": {'article_title': "rabbits"}},
			{"term": {'article_title': "guinea-pig"}},
			{"term": {'article_title': "ovine"}},
			{"term": {'article_title': "postpartum"}},
			{"term": {'article_title': "guinea pig"}},
			{"term": {'article_title': "rats"}}
			]
	return filt

def get_concept_string(conceptid_series):
	result_string = ""
	for item in conceptid_series:
		result_string += item + " "
	
	return result_string.strip()

def get_concept_query_string(full_conceptid_list):

	query_string = ""
	for item in full_conceptid_list:
		if type(item) == list:
			counter = 10
			query_string += "( "
			for concept in item:
				query_string += concept + "^" + str(counter) + " OR "
				if counter != 1:
					counter -= 1
			query_string = query_string.rstrip('OR ')
			query_string += ") AND "
		else:
			query_string += item + " AND "
	query_string = query_string.rstrip("AND ")

	return query_string

def get_article_type_query_string(concept_type_list, unmatched_terms):
	# RCT, meta-analysis, network-meta, cross-over study, case report
	if 'condition' in concept_type_list and ('treatment' in concept_type_list or 'treatment' in unmatched_terms):
		return "( 887729^15 OR 887761^7 OR 887749^7 OR 887770^2 OR 887774^2 OR 887763^1)"
	# prevention procedure not in concept_type
	# Systematic review, meta-analysis
	elif 'condition' in concept_type_list and unmatched_terms == '' and \
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
	elif 'prevention' in concept_type_list or 'prevention' in unmatched_terms or '379416' in concept_type_list:
		return "( 887729^10 OR 887772^5 OR 887774^5 OR 887769^5 )"
	else:
		return ""

def get_query(full_conceptid_list, unmatched_terms, query_types_list, journals, start_year, end_year, fields_arr, cursor):
	es_query = {}
	if unmatched_terms == '':

		es_query["function_score"] = {"query" : { "bool" : {\
							"must_not": get_article_type_filters(), \
							"must": \
								[{"query_string": {"fields" : fields_arr, \
								 "query" : get_concept_query_string(full_conceptid_list)}}], \
							"should": \
								[{"query_string" : {"fields" : fields_arr, 
								"query" : get_article_type_query_string(query_types_list, unmatched_terms)}}]}}, \
								 "functions" : [{"filter" : {"range": {"journal_pub_year": {"gte" : "1990"}}}, "weight" : 1.4}]}

	else:
		# Unmatched terms need to be formatted for lucene

		es_query["function_score"] = {"query" : {"bool" : {\
						"must_not": get_article_type_filters(), \
						"must": \
							[{"query_string": {"fields" : fields_arr, \
							 	"query" : get_concept_query_string(full_conceptid_list)}}, {"query_string": {\
								"query" : get_unmatched_query_string(unmatched_terms)}}],
						"should": \
							[{"query_string" : {"fields" : fields_arr, 
							"query" : get_article_type_query_string(query_types_list, unmatched_terms)}}]
						}}, 
						"functions" : [{"filter" : {"range": {"journal_pub_year": {"gte" : "1990"}}}, "weight" : 1.4}]}

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

		es_query["function_score"]["query"]["bool"]["filter"] = d

	return es_query


def get_unmatched_query_string(unmatched_terms):
	query_string = ""
	unmatched_list = unmatched_terms.rstrip(' ').split(' ')

	for i in unmatched_list:
		query_string += "( " + i + " ) AND "
	query_string = query_string.rstrip("AND ")

	return query_string


def get_text_query(query):
	es_query = {
			"size" : 400,
			 "query": \
			 	{"bool": { \
					"must": \
						{"query_string": {"query" : query}}, \
					"must_not": get_article_type_filters()}}}

	return es_query


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


def get_query_concept_types_df_3(conceptid_df, query_concept_list, cursor):

	dist_concept_list = list(set(conceptid_df['acid'].tolist()))

	if len(dist_concept_list) > 0:

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




def get_related_conceptids(query_concept_list, original_query_concepts_list, flattened_query, query_types_list, unmatched_terms, filters, cursor):

	result_dict = dict()
	es = es_util.get_es_client()
	# size zero means only aggregation results returned

	jj = u.Timer("related agg query")
	es_query = {
				 "size" : 0, \
				 "query": get_query(query_concept_list, unmatched_terms, query_types_list \
				 	,filters['journals'], filters['start_year'], filters['end_year'] \
				 	,["title_cids^10", "abstract_conceptids.*"], cursor), \
				 "aggs" : {'concepts_of_interest' : {"terms" : {"field" : "concepts_of_interest", "size" : 40000}}, 
				 	}}

	sr = es.search(index=INDEX_NAME, body=es_query, request_timeout=100000)
	jj.stop()
	try:
		res = sr['aggregations']['concepts_of_interest']['buckets']
		cids_of_interest_df = pd.DataFrame.from_dict(res)
		cids_of_interest_df.columns = ['acid', 'count']

	except:
		cids_of_interest_df = pd.DataFrame()

	expanded_query_acids = []
	
	if len(cids_of_interest_df) > 0:
		# expanded_query_acids filters out expanded query to those cids in the search results
		# this improves performance of bolding by filtering concept terms that aren't in search results a priori
		expanded_query_acids = cids_of_interest_df[cids_of_interest_df['acid'].isin(flattened_query)].copy()['acid'].tolist()
		cids_of_interest_df = cids_of_interest_df[cids_of_interest_df['acid'].isin(original_query_concepts_list) == False].copy()
	
	if len(expanded_query_acids) == 0:
		expanded_query_acids = flattened_query

	sub_dict = dict()
	sub_dict['treatment'] = []
	sub_dict['diagnostic'] = []
	sub_dict['condition'] = []
	sub_dict['cause'] = []

	if len(cids_of_interest_df.index) > 0:
		ff = u.Timer("get_query_concepts_types_df_3")
		concept_types_df = get_query_concept_types_df_3(cids_of_interest_df, flattened_query, cursor)
		ff.stop()

		gg = u.Timer("actual aggs")
		if len(concept_types_df.index) > 0:
			zz = u.Timer("add_names")
			concept_types_df = ann2.add_names(concept_types_df, cursor)
			zz.stop()
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
		gg.stop()
	return sub_dict['treatment'], sub_dict['diagnostic'], sub_dict['condition'], sub_dict['cause'], expanded_query_acids
	


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
