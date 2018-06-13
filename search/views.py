from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
import snomed_annotator as ann
import utilities.pglib as pg
from nltk.stem.wordnet import WordNetLemmatizer
import utilities.utils as u
import pandas as pd
import utilities.es_utilities as es_util
from django.http import JsonResponse
# Create your views here.

INDEX_NAME='pubmedx1'

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

###

# def vancomycin(request):
# 	return render(request, 'search/vanco_calc.html')

def vc_main(request):
	vc_payload = {}
	if bool(request.POST):
		cursor = pg.return_postgres_cursor()
		username=request.POST['username']
		u.insert_new_vancocalc_user(username, cursor)
	
	user_df = get_vc_users(cursor)
	user_payload = get_user_payload(user_df)

	return render(request, 'search/vc_main.html', {'user_payload' : user_payload})

def vc_cases(request):
	cases_payload = {}
	if bool(request.POST):
		cursor = pg.return_postgres_cursor()
		uid = request.POST['uid']
		username = request.POST['username']
		cases_df = get_vc_cases(cursor, uid)
		cases_payload = get_cases_payload(cases_df)
	return render(request, 'search/vc_cases.html', {'cases_payload' : cases_payload})

def get_vc_users(cursor):
	query="select uid, username from (select uid, effectivetime, \
		row_number () over (partition by uid order by effectivetime desc) as row_num, \
		username, active from vancocalc.users) tb where row_num = 1 and active=1 "
	user_df = pg.return_df_from_query(cursor, query, None, ["uid", "username"])
	return user_df

def get_vc_cases(cursor, uid):
	query="select cid, casename from (select cid, effectivetime, \
		row_number () over (partition by cid order by effectivetime desc) as row_num, \
		active from vancocalc.cases where uid=%s) tb where row_num = 1 and active=1 "
	case_df = pg.return_df_from_query(cursor, query, (uid,), ["cid", "casename"])
	return case_df

def get_cases_payload(cases_df)
	cases_list = []

	for index,case in cases_df.iterrows():
		hit_dict = {}
		hit_dict["casename"] = case["casename"]
		hit_dict["cid"] = case["cid"]
		cases_list.append(hit_dict)

	return cases_list

def get_user_payload(user_df):
	user_list = []

	for index,user in user_df.iterrows():
		hit_dict = {}
		hit_dict["username"] = user["username"]
		hit_dict["uid"] = user["uid"]
		user_list.append(hit_dict)

	return user_list

def vcSubmit(request):
	age = int(request.GET.get('age', None))
	is_female = request.GET.get('is_female', None)
	if is_female == "true":
		is_female = True 
	else:
		is_female = False
	
	height_ft = float(request.GET.get('height_ft', None))
	height_in = float(request.GET.get('height_in', None))
	weight = float(request.GET.get('actualWeight', None))
	creatinine = float(request.GET.get('creatinine', None))
	
	indication = str(request.GET.get('indication'))
	troughTarget = int(request.GET.get('troughTarget'))
	doseType = str(request.GET.get('doseType'))
	comorbid = request.GET.getlist('comorbid[]')

	pt = Patient(age, is_female, height_ft, height_in, weight, creatinine, comorbid)
	print(doseType)
	data = {"crcl" : pt.crcl}
	# if doseType == "loading":
	# 	data.update(returnLoadingDoseForPatient(pt))
	# 	data["doseType"] = "Loading dose"
	# elif doseType == "initialMaintenance":
	# 	data.update(returnInitialMaintenanceDose(pt, troughTarget))
	# 	data["doseType"] = "Initial maintenace dose"
	
	return JsonResponse(data)

def returnLoadingDoseForPatient(pt):
	if (pt.age >= 70):
		dose = 15*pt.weight
	else:
		dose = 20*pt.weight

	if (dose > 2000):
		dose = 2000
	return returnRoundedDose(dose)

def returnInitialMaintenanceDose(pt, troughTarget):
	dose = {}
	dose_ref = Dose(troughTarget)

	if troughTarget == 0:
		if 'crrt' in pt.comorbid:
			dose["dose"] = dose_ref.doseArr[6]
			dose["freq"] = "q12"
		elif 'hd' in pt.comorbid:
			dose["dose"] = dose_ref.doseArr[5]
			dose["freq"] = "post-HD"
		else:
			if pt.crcl >= 70:
				if ((pt.age > 50) or (('esld' in pt.comorbid) or ('chf' in pt.comorbid) or ('dm' in pt.comorbid))):
					dose["dose"] = dose_ref.doseArr[0]
					dose["freq"] = "q12"
				else:
					dose["dose"] = dose_ref.doseArr[1]
					dose["freq"] = "q8"
			elif (pt.crcl >= 40):
				dose["dose"] = dose_ref.doseArr[2]
				dose["freq"] = "q12"
			elif (pt.crcl >= 20):
				dose["dose"] = dose_ref.doseArr[3]
				dose["freq"] = "q24"
			else:
				dose["dose"] = dose_ref.doseArr[4]
				dose["freq"] = "q48"
	else: #trough target is 15-20
		if 'crrt' in pt.comorbid:
			dose["dose"] = dose_ref.doseArr[6]
			dose["freq"] = "q24"
		elif 'hd' in pt.comorbid:
			dose["dose"] = dose_ref.doseArr[5]
			dose["freq"] = "post-HD"
		else:
			if pt.crcl >= 70:
				if ((pt.age > 50) or ('esld' in pt.comorbid) or ('chf' in pt.comorbid) or ('dm' in pt.comorbid)):
					dose["dose"] = dose_ref.doseArr[0]
					dose["freq"] = "q12"
				else:
					dose["dose"] = dose_ref.doseArr[1]
					dose["freq"] = "q8"
			elif pt.crcl >= 40:
				dose["dose"] = dose_ref.doseArr[2]
				dose["freq"] = "q12"
			elif pt.crcl >= 20:
				dose["dose"] = dose_ref.doseArr[3]
				dose["freq"] = "q24"
			else:
				dose["dose"] = dose_ref.doseArr[4]
				dose["freq"] = "single loading dose followed by kintetics"

	dose["dose"] = dose["dose"]*pt.dosingWeight
	dose["dose"] = returnRoundedDose(dose["dose"])

	return dose

def returnNewMaintenanceDoseForPatient(pt, troughTarget, trough):
	vd = 0.7 * pt.dosingWeight
	vancoClearance = pt.crcl * 0.06
	ke = vancoClearance / vd
	halfLife = 0.693/ke
	dose_ref = Dose(troughTarget)



def returnRoundedDose(dose):
	dose = round(dose/250)*250

	if dose > 2000:
		return 2000
	else:
		return dose

def returnTroughTarget(request):
	indication = request.GET.get('indication')
	data = {"trough" : ""}

	## TROUGH TARGET OF 10-15 == 0
	## TROUGH TARGET OF 15-20 == 1

	if indication == "Bacteremia":
		data["trough"] = "1"
	elif indication == "Brain abscess":
		data["trough"] = "1"
	elif indication == "Cellulitis":
		data["trough"] = "0"
	elif indication == "Cystitis":
		data["trough"] = "0"
	elif indication == "Endocarditis":
		data["trough"] = "1"
	elif indication == "Meningitis":
		data["trough"] == "1"
	elif indication == "Neutropenic fever":
		data["trough"] = "0"
	elif indication == "Osteomyelitis":
		data["trough"] = "1"
	elif indication == "Prosthetic joint infection":
		data["trough"] = "1"
	elif indication == "Pulmonary infection":
		data["trough"] = "1"
	elif indication == "Sepsis":
		data["trough"] = "1"
	elif indication == "Skin and soft tissue infections":
		data["trough"] = "0"
	elif indication == "Urinary tract infection":
		data["trough"] = "0"
	elif indication == "Urosepsis":
		data["trough"] = "1"

	return JsonResponse(data)

### ELASTIC SEARCH


def elastic_search_home_page(request):
	return render(request, 'search/elastic_home_page.html')


def post_elastic_search(request):
	query = request.POST['query']
	return HttpResponseRedirect(reverse('search:elastic_search_results', args=(query,)))

def elastic_search_results(request, query):
	es = u.get_es_client()
	es_query = get_text_query(query)
	sr = es.search(index=INDEX_NAME, body=es_query)['hits']['hits']
	sr_payload = get_sr_payload(sr)

	return render(request, 'search/elastic_search_results_page.html', {'sr_payload' : sr_payload, 'query' : query})


### CONCEPT SEARCH

def concept_search_home_page(request):
	return render(request, 'search/concept_search_home_page.html')

def post_concept_search(request):
	params = {}
	try:
		params['journals'] = request.POST.getlist('journals[]')
	except:
		params['journals'] = []

	try:
		if request.POST['start_year'] == '':
			params['start_year'] = None
		else:
			params['start_year'] = request.POST['start_year']
		
	except:
		params['start_year'] = None

	try:
		if request.POST['end_year'] == '':
			params['end_year'] = None 
		else:
			params['end_year'] = request.POST['end_year']
	except:
		params['end_year'] = None

	params['query'] = request.POST['query']
	# params['end_year'] = request.POST ???????????

	query = request.POST['query']
	# return HttpResponseRedirect(reverse('search:concept_search_results', args=(query,)))
	return HttpResponseRedirect(reverse('search:concept_search_results', kwargs=params))

def post_pivot_search(request):
	conceptid1 = request.POST['conceptid1']
	conceptid2 = request.POST['conceptid2']
	term1 = request.POST['term1']
	term2 = request.POST['term2']
	query = term1 + " " + term2
	params = {}
	try:
		params['journals'] = request.POST.getlist('journals[]')
	except:
		params['journals'] = []

	try:
		if request.POST['start_year'] == '':
			params['start_year'] = None
		else:
			params['start_year'] = request.POST['start_year']
		
	except:
		params['start_year'] = None

	try:
		if request.POST['end_year'] == '':
			params['end_year'] = None 
		else:
			params['end_year'] = request.POST['end_year']
	except:
		params['end_year'] = None
	# return HttpResponseRedirect(reverse('search:conceptid_search_results', kwargs={'query' : query, 'conceptid1' : conceptid1, 'conceptid2' : conceptid2}))
	res = conceptid_search_results(request, query, conceptid1, conceptid2, params['journals'], params['start_year'], params['end_year'])
	params.update(res)
	print(params['journals'])
	print("ASDKLJHASLJHDASLJKASDLJK")
	return render(request, 'search/concept_search_results_page.html', params)

### query contains conceptids instead of text
def conceptid_search_results(request, query, conceptid1, conceptid2, journals, start_year, end_year):

	query_concepts_df = pd.DataFrame([conceptid1, conceptid2], columns=['conceptid'])

	es = u.get_es_client()
	cursor = pg.return_postgres_cursor()
	
	full_query_concepts_list = ann.query_expansion(query_concepts_df['conceptid'], cursor)

	query_concepts_dict = get_query_arr_dict(full_query_concepts_list)

	es_query = {"from" : 0, \
				 "size" : 100, \
				 "query": get_query(full_query_concepts_list, None, journals, start_year, end_year, cursor)}

	sr = es.search(index=INDEX_NAME, body=es_query)

	sr_payload = get_sr_payload(sr['hits']['hits'])
	return {'sr_payload' : sr_payload, 'query' : query, 'concepts' : query_concepts_dict, \
		'at_a_glance' : {'related' : None}}
	


def concept_search_results(request):
	journals = request.GET.getlist('journals[]')

	query = request.GET['query']
	start_year = request.GET['start_year']
	end_year = request.GET['end_year']

	es = u.get_es_client()	

	query = ann.clean_text(query)

	cursor = pg.return_postgres_cursor()
	filter_words_query = "select words from annotation.filter_words"
	filter_words_df = pg.return_df_from_query(cursor, filter_words_query, None, ["words"])

	query_concepts_df = ann.annotate_text_not_parallel(query, filter_words_df, cursor, True)

	sr = dict()
	related_dict = dict()
	query_concepts_dict = dict()
	if query_concepts_df is not None:
		unmatched_terms = get_unmatched_terms(query, query_concepts_df, filter_words_df)
		full_query_concepts_list = ann.query_expansion(query_concepts_df['conceptid'], cursor)

		flattened_query = get_flattened_query_concept_list(full_query_concepts_list)
		
		query_concepts = get_query_concept_types_df(query_concepts_df['conceptid'].tolist(), cursor)
		symptom_count = len(query_concepts[query_concepts['concept_type'] == 'symptom'].index)
		condition_count =len(query_concepts[query_concepts['concept_type'] == 'condition'].index)
		query_concept_count = len(query_concepts_df.index)

		# single condition query
		if (symptom_count == 0 and condition_count != 0 and query_concept_count == 1):
			related_dict = get_related_conceptids(full_query_concepts_list, unmatched_terms, cursor, 'condition')
		elif (symptom_count != 0 and condition_count == 0):
			related_dict = get_related_conceptids(full_query_concepts_list, unmatched_terms, cursor, 'symptom')
		else:
			related_dict = None

		query_concepts_dict = get_query_arr_dict(full_query_concepts_list)

		es_query = {"from" : 0, \
				 "size" : 100, \
				 "query": get_query(full_query_concepts_list, unmatched_terms, journals, start_year, end_year, cursor)}
		print(es_query)
		sr = es.search(index=INDEX_NAME, body=es_query)

	###UPDATE QUERY BELOW FOR FILTERS
	else:
		es_query = get_text_query(query)
		sr = es.search(index=INDEX_NAME, body=es_query)

	sr_payload = get_sr_payload(sr['hits']['hits'])

	return render(request, 'search/concept_search_results_page.html', \
		{'sr_payload' : sr_payload, 'query' : query, 'concepts' : query_concepts_dict, \
		'journals': journals, 'start_year' : start_year, 'end_year' : end_year, 'at_a_glance' : {'related' : related_dict}})




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
		elif len((query_concepts_df[(query_concepts_df['term_end_index'] >= index) & (query_concepts_df['term_start_index'] <= index)]).index) > 0:
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

def get_query(full_conceptid_list, unmatched_terms, journals, start_year, end_year, cursor):
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

	
	if (len(journals) > 0) or start_year or end_year:
		d = {"filter" : {"bool" : {}}}

		if len(journals) > 0:
			d["filter"]["bool"]["should"] = []

			for i in journals:
				d["filter"]["bool"]["should"].append({"match" : {'journal_iso_abbrev' : i}})

		if start_year and end_year:
			d["filter"]["bool"]["must"] = [{"range" : {"journal_pub_year" : {"gte" : start_year, "lte" : end_year}}}]
		elif start_year:
			d["filter"]["bool"]["must"] = [{"range" : {"journal_pub_year" : {"gte" : start_year}}}]
		elif end_year:
			d["filter"]["bool"]["must"] = [{"range" : {"journal_pub_year" : {"lte" : end_year}}}]

		es_query["bool"]["filter"] = d["filter"]

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
	concept_type_query_string = "select * from annotation.concept_types where conceptid in %s"

	query_concept_type_df = pg.return_df_from_query(cursor, concept_type_query_string, \
		(tuple(flattened_concept_list),), ["conceptid", "concept_type"])

	return query_concept_type_df

def get_related_conceptids(query_concept_list, unmatched_terms, cursor, query_type):

	result_dict = dict()
	es = u.get_es_client()
	root_concept_name = ""
	root_cid = None
	if query_type == 'symptom' and len(query_concept_list) > 1:
		root_concept_name = "symptom combination"
		if isinstance(query_concept_list[0], list):
			root_cid = query_concept_list[0][0]
		else:
			root_cid = query_concept_list[0]
	else:
		if isinstance(query_concept_list[0], list):
			root_concept_name = u.get_conceptid_name(query_concept_list[0][0], cursor)
			root_cid = query_concept_list[0][0]
		else:
			root_concept_name = u.get_conceptid_name(query_concept_list[0], cursor)
			root_cid = query_concept_list[0]

	if query_type == 'condition':

		treatments_query = 	{"from" : 0, \
				 "size" : 400, \
				 "query": \
			 		{"bool": { \
						"must": \
							[{"query_string": {"fields" : ["title_conceptids", "abstract_conceptids"], \
							 "query" : get_concept_query_string(query_concept_list, cursor)}}], \
						"must_not": [get_article_type_filters(), {"query_string" : {"fields" : ["title_conceptids"],\
							"query" : '30207005'}}]}}}
		sr = es.search(index=INDEX_NAME, body=treatments_query)

		sr_conceptids = get_conceptids_from_sr(sr)

		if len(sr_conceptids) > 0:

			dist_sr_conceptids = list(set(sr_conceptids))

			agg_df = get_query_concept_types_df(dist_sr_conceptids, cursor)

			agg_df = agg_df[agg_df['concept_type'].isin(['treatment', 'diagnostic'])]

			concept_types = list(set(agg_df['concept_type'].tolist()))

			sub_dict = dict()
			sub_dict['term'] = root_concept_name
			sub_dict['treatment'] = []
			sub_dict['diagnostic'] = []
	
			for concept_type in concept_types:
				if concept_type == 'treatment':
					sr_conceptid_df = agg_df[agg_df['concept_type'] == concept_type].copy()
					sr_conceptid_df['count'] = 1
					sr_conceptid_df = sr_conceptid_df.groupby(['conceptid'], as_index=False)['count'].sum()

					if len(sr_conceptid_df.index) > 0:
							
						sr_conceptid_df = de_dupe_synonyms(sr_conceptid_df, cursor)
						sr_conceptid_df = ann.add_names(sr_conceptid_df)
						sr_conceptid_df = sr_conceptid_df.sort_values(['count'], ascending=False)

						for index,row in sr_conceptid_df.iterrows():
							item_dict = {'conceptid' : row['conceptid'], 'term' : row['term'], 'count' : row['count']}
							sub_dict['treatment'].append(item_dict)
						
				if concept_type == 'diagnostic':
					sr_conceptid_df = agg_df[agg_df['concept_type'] == concept_type].copy()
					sr_conceptid_df['count'] = 1
					sr_conceptid_df = sr_conceptid_df.groupby(['conceptid'], as_index=False)['count'].sum()

					if len(sr_conceptid_df.index) > 0:
						sr_conceptid_df = de_dupe_synonyms(sr_conceptid_df, cursor)
						sr_conceptid_df = ann.add_names(sr_conceptid_df)
						sr_conceptid_df = sr_conceptid_df.sort_values(['count'], ascending=False)
						for index,row in sr_conceptid_df.iterrows():
							item_dict = {'conceptid' : row['conceptid'], 'term' : row['term'], 'count' : row['count']}
							sub_dict['diagnostic'].append(item_dict)

			result_dict[root_cid] = sub_dict
	elif query_type == 'symptom':
		# condition_query =  	{"from" : 0, \
		# 		 "size" : 1000, \
		# 		 "query": \
		# 	 		{"bool": { \
		# 				"must": \
		# 					[{"query_string": {"fields" : ["title_conceptids", "abstract_conceptids"], \
		# 					 "query" : get_concept_query_string(query_concept_list, cursor)}}], \
		# 				"must_not": [get_article_type_filters()]}}}
		condition_query = { "from" : 0, "size" : 400, \
						"query": {"bool" : {"must": \
							[{"query_string": {"fields" : ["title_conceptids^5", "abstract_conceptids.*"], \
							 "query" : get_concept_query_string(query_concept_list, cursor)}}], "must_not" : get_article_type_filters()}}}
						
		# condition_query = {"from" : 0, \
		# 		 "size" : 100, \
		# 		 "query": get_query(query_concept_list, None, cursor)}

		sr = es.search(index=INDEX_NAME, body=condition_query)

		sr_conceptids = get_conceptids_from_sr(sr)

		if len(sr_conceptids) > 0:

			dist_sr_conceptids = list(set(sr_conceptids))

			agg_df = get_query_concept_types_df(dist_sr_conceptids, cursor)
			agg_df = agg_df[agg_df['concept_type'].isin(['condition'])]

			concept_types = list(set(agg_df['concept_type'].tolist()))

			sub_dict = dict()
			sub_dict['term'] = root_concept_name
			sub_dict['condition'] = []

			sr_conceptid_df = agg_df.copy()
			sr_conceptid_df['count'] = 1
			sr_conceptid_df = sr_conceptid_df.groupby(['conceptid'], as_index=False)['count'].sum()

			if len(sr_conceptid_df) > 0:
							
				sr_conceptid_df = de_dupe_synonyms(sr_conceptid_df, cursor)
				sr_conceptid_df = ann.add_names(sr_conceptid_df)
				sr_conceptid_df = sr_conceptid_df.sort_values(['count'], ascending=False)

				for index,row in sr_conceptid_df.iterrows():
					item_dict = {'conceptid' : row['conceptid'], 'term' : row['term'], 'count' : row['count']}
					sub_dict['condition'].append(item_dict)
			result_dict[root_cid] = sub_dict


	return result_dict

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

def get_conceptids_from_sr(sr):
	conceptid_list = []

	for hit in sr['hits']['hits']:
		if hit['_source']['title_conceptids'] is not None:
			conceptid_list.extend(hit['_source']['title_conceptids'])
		if hit['_source']['abstract_conceptids'] is not None:
			for key1 in hit['_source']['abstract_conceptids']:
				for key2 in hit['_source']['abstract_conceptids'][key1]:
					if hit['_source']['abstract_conceptids'][key1][key2] is not None:
						conceptid_list.extend(hit['_source']['abstract_conceptids'][key1][key2])
	return conceptid_list


############################### VANCO CALC FUNCTIONS

def returnIdealBodyWeight(is_female, height):
	if not is_female:
		return (50+(2.3*(height-60)))
	else:
		return (45.5 + (2.3 * (height-60)))

def returnAdjustedWeightForIdealBodyWeight(idealBodyWeight, actualWeight):
	return (idealBodyWeight + (0.4 * (actualWeight - idealBodyWeight)))

def returnDosingWeight(pt):
	idealBodyWeight = returnIdealBodyWeight(pt.is_female, pt.height)
	print("idealBodyWeight")
	print(idealBodyWeight)
	print("endIdealBodyWeight")
	if (pt.weight > (1.2*idealBodyWeight)):
		return returnAdjustedWeightForIdealBodyWeight(idealBodyWeight, pt.weight)
	elif (pt.weight < idealBodyWeight):
		return pt.weight
	else:
		return pt.weight

def returnCrCl(pt):

	dosingWeight = pt.setDosingWeight()

	if ((pt.age > 65) and (pt.creatinine < 1)):
		pt.creatinine = 1

	if not pt.is_female:
		return round(((140-pt.age)*dosingWeight)/(72*pt.creatinine), 2)
	else:
		return round(((140-pt.age)*dosingWeight*0.85)/(72*pt.creatinine), 2)


class Patient:
	def __init__(self, age, is_female, height_ft, height_in, weight, creatinine, comorbid):
		self.age = age 
		self.is_female = is_female
		self.height = height_in + (12*height_ft) #height stored in inches
		self.weight = weight
		self.creatinine = creatinine
		self.comorbid = comorbid
		self.dosingWeight = self.setDosingWeight()
		self.crcl = self.setCrCl()

	def setDosingWeight(self):
		return returnDosingWeight(self)

	def setCrCl(self):
		return returnCrCl(self)

	def getCrCl(self):
		return self.crcl

class Dose:
	def __init__(self, troughTarget):
		self.troughTarget = troughTarget

		if troughTarget == 0:
			self.doseArr = [15,12,12,12,12,7,10]
		elif troughTarget == 1:
			self.doseArr = [18,15,15,15,0,7,15]

		self.freqArr = ["8", "12", "12", "24", "0", "post-HD", "24"]


