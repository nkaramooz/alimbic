from django.shortcuts import render
from django.views.decorators.csrf import requires_csrf_token
from django.http import HttpResponseRedirect, HttpResponse
from django.utils.safestring import mark_safe
from snomed_annotator import snomed_annotator as ann
import utilities.pglib as pg
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import utilities.es_utilities as es
from utilities.query import Query
from django.http import JsonResponse
import json
import urllib.parse as urlparse
from urllib.parse import parse_qs
import re
import os


INDEX_NAME=os.environ["INDEX_NAME"]

# Initialize lemmatizer
lmtzr = WordNetLemmatizer()


def home(request):
	return render(request, 'search/concept_search_home_page.html', 
			{'sr_payload' : None, 'query' : '', 'query_annotation' : None, 'unmatched_list' : None, 'concepts' : None, 'primary_cids' : None,
				'journals': None, 'start_year' : '', 'end_year' : '', 'at_a_glance' : {'related' : None}, \
				'treatment' : None, 'diagnostic' : None, 'cause' : None, 'condition' : None, \
				'calcs' : None}
			)


def about(request):
	return render(request, 'about/about.html')


def terms(request):
	return render(request, 'about/terms.html')
	

# Main function for the search page.
@requires_csrf_token
def post_search_text(request):
	spellcheck_threshold = 100
	filters = get_query_filters(request)

	if request.method == 'POST':
		data = json.loads(request.body)
		query_string = data['query']
		query_type = data['query_type']
		
		
		# List of words in the query that did not match to a concept.
		unmatched_terms_list = data['unmatched_terms_list'] if 'unmatched_terms_list' in data else []
		
		# List of concept ids that matched to the query directly.
		nested_query_acids = data['nested_query_acids'] if 'nested_query_acids' in data else []
		
		# Generate the flat query acids list if nested has elements.
		flat_query_acids = [acid for item in nested_query_acids for acid in item]

		# Pivot concept is populated when the user selects a pivot concept from their search results.
		pivot_concept = data['pivot_concept'] if 'pivot_concept' in data else None
		pivot_term = data['pivot_term'] if 'pivot_term' in data else None

		# History stores the log of concepts previously selected through the pivot action.
		pivot_concept_history = data['pivot_history'] if 'pivot_history' in data else []
	# If GET request, this is from a link and requires parsing the parameters in the URL.
	elif request.method == 'GET': 
		parsed = urlparse.urlparse(request.path)
		parsed = parse_qs(parsed.path)
		query = parsed['/search/query'][0]

		
		nested_query_acids = parsed['nested_query_acids[]'] if 'nested_query_acids[]' in parsed else []
		pivot_concept_history = parsed['pivot_history[]'] if 'pivot_history[]' in parsed else []
		unmatched_terms_list = parsed['unmatched_list[]'] if 'unmatched_list[]' in parsed else []
		
		pivot_concept = parsed['pivot_cid'] if 'pivot_cid' in parsed else None
		pivot_term = parsed['pivot_term'][0] if 'pivot_term' in data else None

		if 'query_type' in parsed:
			query_type = parsed['query_type'][0]

	if query_type == 'keyword':
		cleaned_query_string, query_concepts_df = get_query_annotation_features(query_string, spellcheck_threshold)

		# Flat list of unique acids.
		flat_query_acids = list(set(query_concepts_df['acid'].dropna().tolist()))

		# Concept search
		if len(flat_query_acids) > 0:
			# Format of [[],[]]. Items in the inner array will be interpreted as "AND"
			# Items across the outer array will be interpreted as "OR".
			nested_query_acids = [[acid] for acid in flat_query_acids]

			# in the event of keyword search after pivot
			# pivot history will be out of sync (ex deleting pivot term)
			pivot_concept_history = list(set(pivot_concept_history).intersection(set(flat_query_acids)))
			query_concept_types_list = get_query_concept_types_df(flat_query_acids)['concept_type'].tolist()
			unmatched_terms_list = get_unmatched_list(cleaned_query_string, query_concepts_df)
			
			# Format similar to the nested_query_acids [[], []], []].
			# Query expansion acids are added to the inner array which gets translated to "OR" statements
			# for elastic search.
			nested_expanded_query_acids = ann.query_expansion(nested_query_acids)
			flat_expanded_query_acids = [a_cid for item in nested_expanded_query_acids for a_cid in item]

			query_obj = Query(
				query_string = query_string,
				cleaned_query_string = cleaned_query_string,
				nested_query_acids = nested_query_acids,
				flat_query_acids = flat_query_acids,
				nested_expanded_query_acids = nested_expanded_query_acids,
				flat_expanded_query_acids = flat_expanded_query_acids,
				unmatched_terms_list = unmatched_terms_list,
				filters = filters,
				query_concept_types_list = query_concept_types_list,
				pivot_history = pivot_concept_history
				)
		else:
			unmatched_terms_list = cleaned_query_string.split(' ')
			nested_expanded_query_acids = []
			query_obj = Query(
				query_string = query_string,
				cleaned_query_string = cleaned_query_string,
				unmatched_terms_list = unmatched_terms_list,
				filters = filters,
				pivot_history = pivot_concept_history
				)
	
		query_obj.es_query = get_query(query_obj)
		
		es_query = {"from" : 0, \
			 "size" : 20, \
			 "query": query_obj.es_query}
		
		sr = es.search(es_query, INDEX_NAME)

	elif query_type == 'pivot':
		cleaned_query_string = ann.clean_text(query_string)
		flat_query_acids.append(pivot_concept)
		nested_query_acids.append([pivot_concept])
		pivot_concept_history.append(pivot_concept)

		query_concept_types_list = get_query_concept_types_df(flat_query_acids)['concept_type'].tolist()

		nested_expanded_query_acids = ann.query_expansion(nested_query_acids)
		flat_expanded_query_acids = [a_cid for item in nested_expanded_query_acids for a_cid in item]
		
		query_obj = Query(
			query_string = query_string,
			cleaned_query_string = cleaned_query_string,
			nested_query_acids = nested_query_acids,
			flat_query_acids = flat_query_acids,
			nested_expanded_query_acids = nested_expanded_query_acids,
			flat_expanded_query_acids = flat_expanded_query_acids,
			unmatched_terms_list = unmatched_terms_list,
			filters = filters,
			query_concept_types_list = query_concept_types_list,
			pivot_history = pivot_concept_history,
			pivot_term = pivot_term
			)
		query_obj.es_query = get_query(query_obj)
		es_query = {"from" : 0, "size" : 20, "query": query_obj.es_query}
		sr = es.search(es_query, INDEX_NAME)

	sr_payload = get_sr_payload(sr['hits']['hits'], query_obj)
	pivots_df = get_pivot_concepts(query_obj) # May be an empty data frame
	treatment_dict = rollups(pivots_df[pivots_df['concept_type'] == 'treatment'])
	diagnostic_dict = rollups(pivots_df[pivots_df['concept_type'] == 'diagnostic'])
	cause_dict = rollups(pivots_df[pivots_df['concept_type'] == 'cause'])
	condition_dict = rollups(pivots_df[pivots_df['concept_type'] == 'condition'])
	calc_dict = get_calcs(query_obj.flat_expanded_query_acids)
	
	return get_sr_html(request=request, 
					  query_obj=query_obj,
					  sr_payload=sr_payload, 
					  filters=filters, 
					  treatment_dict=treatment_dict, 
					  diagnostic_dict=diagnostic_dict, 
					  cause_dict=cause_dict, 
					  condition_dict=condition_dict, 
					  calc_dict=calc_dict)	


# Returns the html for the search results based on the parameters provided. 
def get_sr_html(request, query_obj, sr_payload, filters, treatment_dict, diagnostic_dict, cause_dict, condition_dict, calc_dict):
	html = render(request, \
			   'search/concept_search_results_page.html',\
				{'sr_payload' : sr_payload, 
				'query_obj' : query_obj.return_json(),
				'primary_a_cids' : query_obj.flat_query_acids,
				'unmatched_list' : query_obj.unmatched_terms_list,
				'pivot_history' : query_obj.pivot_history,
				'journals': filters['journals'], 'start_year' : filters['start_year'], 'end_year' : filters['end_year'], \
				'treatment' : treatment_dict, 'diagnostic' : diagnostic_dict, 'cause' : cause_dict, 'condition' : condition_dict, \
				'calcs' : calc_dict})
	return html


# Returns the calculators associated with the concepts in the query.
def get_calcs(concepts):
	if len(concepts) > 0:
		query = """
			select distinct on (title, t1.description, url) title, t1.description, url 
			from annotation2.mdc_final t1 where acid in %s 
		"""
		calcs = pg.return_df_from_query(query, (tuple(concepts),), ['title', 'desc', 'url'])

		calc_json = []
		calcs_dict = calcs.to_dict('records')
		for row in calcs_dict:
			calc_json.append({'title' : row['title'], 'desc' : row['desc'], 'url' : row['url']})
		return calc_json
	else:
		return None


# This function takes in a data frame of concept ids, terms and 
# groups them into a parent child relationship.
# This is used for pivot concepts so that rather than showing a lengthy list
# of related concepts (ex. would be "heart failure", "heart failure with reduced ejection fraction", etc)
# the concepts are grouped for display.
# TODO: Need to not show pivots that are parents of the concepts in the query.
def rollups(cids_df):
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
		parents_df = pg.return_df_from_query(query, params, ["child_acid", "parent_acid"])

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


# TODO: Implement logging
def log_query (ip_address, query, primary_cids, unmatched_list, filters):
	insert_query = """
		insert into search.query_logs
		VALUES(%s, %s, %s, %s, %s,%s, %s, now())
	"""
	pg.write_data(insert_query, (ip_address, query,json.dumps(primary_cids), \
		unmatched_list,filters['start_year'], filters['end_year'], json.dumps(filters['journals'])))


# Get the IP address from the request.
def get_ip_address(request):
	ip = ''
	x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
	if x_forwarded_for:
		ip = x_forwarded_for.split(',')[0]
	else:
		ip = request.META.get('REMOTE_ADDR')
	return ip


### Utility functions

# This function takes the list of search results
# and bolds the terms that match the expanded query list.
# For example, if someone searches for "Heart Failure" then the values "CHF"
# and "Heart failure with reduced ejection fraction" will also be bolded since these
# are child concepts of "Heart Failure".
# Unmatched terms (not tied to a concept) will also be bolded.
def get_sr_payload(sr, query_obj):
	sr_list = []
	terms = []
		
	if len(query_obj.flat_expanded_query_acids) > 0:
		query = """
			select term from annotation2.used_descriptions where acid in %s order by length(term) desc
		"""
		terms = pg.return_df_from_query(query, (tuple(query_obj.flat_expanded_query_acids),), ["term"])["term"].tolist()

	terms.extend(query_obj.unmatched_terms_list)

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


# Defines the component of the abstract that is shown in the preview.
def get_show_hide_components(sr_src, hit_dict):
	if sr_src['article_abstract'] is not None:
		top_key_list = list(sr_src['article_abstract'].keys())
		top_key = top_key_list[0]

		hit_dict['abstract_show'] = (('abstract' if top_key == 'unassigned' else top_key),
			sr_src['article_abstract'][top_key])
		hit_dict['abstract_hide'] = list()
		for key1,value1 in sr_src['article_abstract'].items():
			hit_dict['abstract_hide'].append((key1, value1)) if key1 != top_key else None
		hit_dict['abstract_hide'] = (None if len(hit_dict['abstract_hide']) == 0 else hit_dict['abstract_hide'])
	else:
		hit_dict['abstract_show'] = None
		hit_dict['abstract_hide'] = None
	return hit_dict


# Returns the cleaned query string and the concepts dataframe annotated to the query.
def get_query_annotation_features(query_string, spellcheck_threshold):
	cleaned_query_string = ann.clean_text(query_string)
	words = ann.get_all_words_list(cleaned_query_string)
	cache = ann.get_cache(all_words_list=words, case_sensitive=False, \
			check_pos=False, spellcheck_threshold=spellcheck_threshold, lmtzr=lmtzr)
		
		# Get the query in the form of a dataframe that can be used for annotation.
	query_df = ann.return_section_sentences(cleaned_query_string, 'query', 0, pd.DataFrame())
	query_concepts_df = ann.annotate_text(sentences_df=query_df, cache=cache, \
		case_sensitive=False, check_pos=False, acr_check=False,\
		spellcheck_threshold=spellcheck_threshold, \
		return_details=False, lmtzr=lmtzr)
	query_concepts_df.rename(columns={'a_cid' : 'acid'}, inplace=True)

	return cleaned_query_string, query_concepts_df


# Returns a list of terms that did not get matched to a concept.
def get_unmatched_list(query, query_concepts_df):
	unmatched_list = []
	for index,word in enumerate(query.split()):
		if not query_concepts_df.empty and len((query_concepts_df[(query_concepts_df['description_end_index'] >= index) \
			& (query_concepts_df['description_start_index'] <= index)]).index) > 0:
			continue
		else:
			unmatched_list.append(word)
	return unmatched_list


# Additional filters applied to articles to provide more
# clinically meaningful search results.
def get_article_type_filters():
	filters = [ \
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
	return filters


# Returns the elastic search query string for the conceptids in the query.
def get_concept_query_string(nested_expanded_query_acids):
	query_string = ""
	for inner_list in nested_expanded_query_acids:
		query_string += "( "
		
		counter = 10
		for concept in inner_list:
			query_string += concept + "^" + str(counter) + " OR "
			if counter != 1:
				counter -= 1
		query_string = query_string.rstrip('OR ')
		query_string += ") AND " 

	query_string = query_string.rstrip("AND ")
	return query_string


# Using the concept types in the query, this function returns
# the weights for the relevant article types.
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


# Returns the elastic search query string.
# nested_expanded_query_acids is the 2-d concept array. Items within the inner array
# are interpreted as OR statements. Across the outer array, items are interpreted as AND statements.
# The unmatched tokens are included as "should" statements in the elastic query, unless
# no concepts have been annotated to the query, in which case, the terms of the query
# are required to be there ("must" statement).
# Unmatched terms were only be searched on the abstract body.
def get_query(query_obj, fields_arr=["title_cids^10", "abstract_conceptids.*"]):
	es_query = {}
	# Search only occurs across the concept space.
	if query_obj.nested_expanded_query_acids is not None and len(query_obj.unmatched_terms_list) == 0:
		es_query["function_score"] = {"query" : { "bool" : {
							"must_not": get_article_type_filters(), 
							"must": 
								[{"query_string": {"fields" : fields_arr, 
								 "query" : get_concept_query_string(query_obj.nested_expanded_query_acids)}}], 
							"should": 
								[{"query_string" : {"fields" : fields_arr, 
								"query" : get_article_type_query_string(query_obj.query_concept_types_list, query_obj.unmatched_terms_list)}}]}}, 
							"functions" : []}
	# Search occurrs across the concept space and free text.
	elif query_obj.nested_expanded_query_acids is not None and len(query_obj.unmatched_terms_list) > 0:
		es_query["function_score"] = {
				"query" : {
					"bool" : {
						"must_not" : get_article_type_filters(), 
						"must" : [
							{"query_string": {"fields" : fields_arr, 
						 					"query" : get_concept_query_string(query_obj.nested_expanded_query_acids)}}, 
							{"query_string": { "fields" : ["article_abstract.*"],
											"query" : get_unmatched_query_string(query_obj.unmatched_terms_list)}}
							],
						"should" : 
							[{"query_string" : {"fields" : fields_arr, 
											"query" : get_article_type_query_string(query_obj.query_concept_types_list, query_obj.unmatched_terms_list)}}
												]
						}}, 
				"functions" : []}
	# Search only across free text.
	else:
		es_query["function_score"] = {
			"query" : { 
				"bool" : {
					"must_not": get_article_type_filters(), 
					"must": {
						"query_string" : { 
							"query" : get_unmatched_query_string(query_obj.unmatched_terms_list)
							}}}},
			"functions" : []}
	
	if (len(query_obj.filters['journals']) > 0) or query_obj.filters['start_year'] or query_obj.filters['end_year']:
		d = []

		if len(query_obj.filters['journals']) > 0:

			for i in query_obj.filters['journals']:
				d.append({"term" : {'journal_iso_abbrev' : i}})

		if query_obj.filters['start_year'] and query_obj.filters['end_year']:
			d.append({"range" : {"journal_pub_year" : {"gte" : query_obj.filters['start_year'], "lte" : query_obj.filters['end_year']}}})
		elif query_obj.filters['start_year']:
			d.append({"range" : {"journal_pub_year": {"gte" : query_obj.filters['start_year']}}})
		elif query_obj.filters['end_year']:
			d.append({"range" : {"journal_pub_year" : {"lte" : query_obj.filters['end_year']}}})

		es_query["function_score"]["query"]["bool"]["filter"] = d

	if not query_obj.filters['start_year']:
		es_query["function_score"]["functions"].append(\
			{"filter" : {"range": {"journal_pub_year": {"gte" : "1990"}}}, "weight" : 1.4})

	return es_query


# Returns the elastic search query string for a list of terms
# that were unmatched to concepts. Unmatched terms are all included
# with an "AND" relationship.
def get_unmatched_query_string(unmatched_list):
	query_string = ""
	for i in unmatched_list:
		query_string += "( " + i + " ) AND "
	query_string = query_string.rstrip("AND ")

	return query_string


# Returns a dataframe with the conceptid and concept type.
def get_query_concept_types_df(flattened_concept_list):
	concept_type_query_string = """
		select root_acid as acid, rel_type as concept_type 
		from annotation2.concept_types where active=1 and root_acid in %s
	"""
	query_concept_type_df = pg.return_df_from_query(concept_type_query_string, \
		(tuple(flattened_concept_list),), ["acid", "concept_type"])
	return query_concept_type_df


# Function takes in a query object and returns a dictionary
# with keys corresponding to the main concept types (treatment, diagnostic, condition, cause).
# Values in the dictionary are lists of concept ids. Returns an empty data frame if no pivots were found.
def get_pivot_concepts(query_obj):
	pivots_df = pd.DataFrame(columns=['acid', 'concept_type'])
	es_agg_query = {
				 "size" : 0, \
				 "query": query_obj.es_query, \
				 "aggs" : {'concepts_of_interest' : {"terms" : {"field" : "concepts_of_interest", "size" : 40000}}, 
			}}
	sr = es.search(es_agg_query, INDEX_NAME)
	sr_interesting_concepts_df = pd.DataFrame.from_dict(sr['aggregations']['concepts_of_interest']['buckets'])
	
	# Of the interesting concepts, exclude concepts that were in the original query (not the expanded query)
	if query_obj.flat_query_acids is not None and len(sr_interesting_concepts_df.index) > 0:
		sr_interesting_concepts_df.rename(columns={'key' : 'acid', 'doc_count' : 'count'}, inplace=True)
		sr_interesting_concepts_df['count'] = sr_interesting_concepts_df.groupby('acid')['count'].transform('sum')
		sr_interesting_concepts_df = sr_interesting_concepts_df[~sr_interesting_concepts_df['acid'].isin(\
			query_obj.flat_query_acids)].copy()
	
	# If there are still interesting concepts, get the concept types
	if len(sr_interesting_concepts_df.index) > 0:
		pivots_df = get_pivot_concept_types(sr_interesting_concepts_df, query_obj)

	return pivots_df


# Given concepts in the search results, this function will return the categorized
# pivot concepts related to the query concepts.
# The returned data frame includes the acid, concept_type, and term. 
# Term is the preferred name of the acid.
def get_pivot_concept_types(sr_interesting_concepts_df, query_obj):
	pivots_df = pd.DataFrame(columns=['acid', 'concept_type'])
	distinct_interesting_concepts_list = list(set(sr_interesting_concepts_df['acid'].tolist()))
	if len(distinct_interesting_concepts_list) > 0:
		
		# Pivot concepts will leverage the expanded query concepts
		if len(query_obj.flat_expanded_query_acids) > 0:
			treatment_query = """
				select 
					distinct(treatment_acid) as acid
					,'treatment' as concept_type
				from ml2.treatment_recs_final_1
				where condition_acid in %s and treatment_acid in %s
					and treatment_acid in
					(select root_acid from annotation2.concept_types where active=1 and rel_type='treatment')
			"""
			treatment_df = pg.return_df_from_query(treatment_query, \
					(tuple(query_obj.flat_expanded_query_acids), tuple(distinct_interesting_concepts_list)),\
					["acid", "concept_type"])
			
			pivots_df = pd.concat([pivots_df, treatment_df], ignore_index=True)

			pivot_types = ['cause', 'diagnostic', 'condition']
			pivot_query = """
					select 
						root_acid as acid
						,rel_type as concept_type
					from annotation2.concept_types
					where active=1 and rel_type in %s and root_acid in %s
			"""
			other_pivot_types_df = pg.return_df_from_query(pivot_query, \
				(tuple(pivot_types), tuple(distinct_interesting_concepts_list)), ["acid", "concept_type"])
			
			pivots_df = pd.concat([pivots_df, other_pivot_types_df], ignore_index=True)
			
		# If no concepts were tagged in the query, then the concept types in the search results
		# are used as the pivot concepts.
		else:
			pivot_query = """
				select 
					root_acid as acid
					,rel_type as concept_type
				from annotation2.concept_types
				where active=1 and root_acid in %s
			"""
			query_concept_type_df = pg.return_df_from_query(pivot_query, \
				(tuple(distinct_interesting_concepts_list),), ["acid", "concept_type"])
			pivots_df = pd.concat([pivots_df, query_concept_type_df], ignore_index=True)
	pivots_df = ann.add_names(pivots_df)
	pivots_df = pivots_df.merge(sr_interesting_concepts_df, how='inner', on=['acid'])
	return pivots_df


# Returns a dictionary with the query filters that include the list of journals
# start year and end year.
def get_query_filters(request):
	filters = {}
	if request.method == 'POST':
		data = json.loads(request.body)
		
		filters['journals'] = data['journals'] if 'journals' in data else []
		if 'start_year' in data and data['start_year'] != '':
			filters['start_year'] = int(data['start_year'])
		else:
			filters['start_year'] = ''

		if 'end_year' in data and data['end_year'] != '':
			filters['end_year'] = int(data['end_year'])
		else:
			filters['end_year'] = ''
	# Get request requires parsing URL
	else: 
		parsed = urlparse.urlparse(request.path)
		parsed = parse_qs(parsed.path)
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
	return filters

