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

def vc_new_case(request):
	new_case_payload = {}
	new_case_payload['uid'] = request.GET['uid']
	new_case_payload['username'] = request.GET['username']
	return render(request, 'search/new_case.html', {'new_case_payload' : new_case_payload})

def vc_case_view(request):
	print(request)
	case_payload = {}
	cursor = pg.return_postgres_cursor()
	cid = ""
	casename=""
	username=""
	case = None
	uid = ""

	if request.method == "GET":
		uid = request.GET['uid']
		username = request.GET['username']
		casename = request.GET['casename']
		cid = request.GET['cid']
		case = Case(cid=cid, cursor=cursor)
	elif 'create' in request.POST:
		cursor = pg.return_postgres_cursor()
		casename = request.POST.get('casename')
		uid = request.POST.get('uid')
		username=request.POST.get('username')
		cid = u.insert_new_vc_case(uid, casename, cursor)

		age = int(request.POST.get('age'))
		is_female = request.POST.get('is_female', None)
		if is_female == "true":
			is_female = 1 
		else:
			is_female = 0
		
		height = 12*float(request.POST.get('height-ft')) + float(request.POST.get('height-in'))
		weight = float(request.POST.get('weight'))
		creatinine = float(request.POST.get('creatinine'))
		bl_creatinine = float(request.POST.get('bl_creatinine'))
		indication = str(request.POST.get('indication'))
		targetTrough = int(request.POST.get('targetTrough'))

		comorbid = request.POST.getlist('comorbid[]')
		
		case = Case(age=age, is_female=is_female, height=height, weight=weight, creatinine=creatinine, bl_creatinine=bl_creatinine, comorbid=comorbid)
		case.save(cid, cursor)
	elif 'addCr' in request.POST:
		uid = request.POST['uid']
		username = request.POST['username']
		casename = request.POST['casename']
		cid = request.POST['cid']
		creatinine = request.POST['cr']
		case = Case(cid=cid, cursor=cursor)
		case.addCr(creatinine)
		case.save(cid, cursor)


		# pt.writeDB(cid, cursor)
	
	case_payload['uid'] = uid
	case_payload['username'] = username
	case_payload['casename'] = casename
	case_payload['cid'] = cid

	case_payload = get_case_payload(case, case_payload)

	return render(request, 'search/case_view.html', {'case_payload' : case_payload})

def get_case_payload(case, case_payload):
	case_payload["chf"] = case.chf.value
	case_payload["dm"] = case.dm.value
	case_payload["esld"] = case.esld.value
	case_payload["crrt"] = case.crrt.value
	case_payload["hd"] = case.hd.value
	case_payload["creatinine"] = case.getCrDict()
	case_payload["weight"] = case.getWeightDict()

	return case_payload

def vc_dosing(request):
	return render(request, 'search/vanco_calc.html')

def vc_main(request):
	vc_payload = {}
	cursor = pg.return_postgres_cursor()
	if request.method == "POST":
		username=request.POST['username']
		u.insert_new_vc_user(username, cursor)
	
	user_df = get_vc_users(cursor)
	user_payload = get_user_payload(user_df)

	return render(request, 'search/vc_main.html', {'user_payload' : user_payload})

def vc_cases(request):
	cases_payload = {}
	cursor = pg.return_postgres_cursor()

	uid = ""
	username = ""
	cases_df = None

	if request.method == "POST":
		uid = request.POST['uid']
		username = request.POST['username']
		casename = request.POST['casename']
		u.insert_new_vc_case(uid, casename, cursor)
	else:
		uid = request.GET['uid']
		username = request.GET['username']
	
	cases_df = get_vc_cases(cursor, uid)
	cases_payload['cases'] = get_cases_payload(cases_df)
	cases_payload['uid'] = uid
	cases_payload['username'] = username


	return render(request, 'search/vc_cases.html', {'cases_payload' : cases_payload})

def get_vc_users(cursor):
	query="select uid, username from (select uid, effectivetime, \
		row_number () over (partition by uid order by effectivetime desc) as row_num, \
		username, active from vancocalc.users) tb where row_num = 1 and active=1 "
	user_df = pg.return_df_from_query(cursor, query, None, ["uid", "username"])
	return user_df

def get_vc_cases(cursor, uid):
	query="select cid, casename from (select cid, casename, effectivetime, \
		row_number () over (partition by cid) as row_num, \
		active from vancocalc.cases where uid=%s) tb where row_num = 1 and active=1 order by effectivetime desc "
	case_df = pg.return_df_from_query(cursor, query, (uid,), ["cid", "casename"])
	return case_df

def vc_loading(request):
	print("true")

def vc_maintenance(request):
	print("true")

def vc_redose(request):
	print('true')

def get_cases_payload(cases_df):
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

	data = {"crcl" : pt.crcl[-1]}
	# if doseType == "loading":
	# 	data.update(returnLoadingDoseForPatient(pt))
	# 	data["doseType"] = "Loading dose"
	# elif doseType == "initialMaintenance":
	# 	data.update(returnInitialMaintenanceDose(pt, troughTarget))
	# 	data["doseType"] = "Initial maintenace dose"
	
	return JsonResponse(data)

def returnLoadingDoseForPatient(pt):
	if (pt.age >= 70):
		dose = 15*pt.weight[-1]
	else:
		dose = 20*pt.weight[-1]

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
			if pt.crcl[-1] >= 70:
				if ((pt.age > 50) or (('esld' in pt.comorbid) or ('chf' in pt.comorbid) or ('dm' in pt.comorbid))):
					dose["dose"] = dose_ref.doseArr[0]
					dose["freq"] = "q12"
				else:
					dose["dose"] = dose_ref.doseArr[1]
					dose["freq"] = "q8"
			elif (pt.crcl[-1] >= 40):
				dose["dose"] = dose_ref.doseArr[2]
				dose["freq"] = "q12"
			elif (pt.crcl[-1] >= 20):
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
			if pt.crcl[-1] >= 70:
				if ((pt.age > 50) or ('esld' in pt.comorbid) or ('chf' in pt.comorbid) or ('dm' in pt.comorbid)):
					dose["dose"] = dose_ref.doseArr[0]
					dose["freq"] = "q12"
				else:
					dose["dose"] = dose_ref.doseArr[1]
					dose["freq"] = "q8"
			elif pt.crcl[-1] >= 40:
				dose["dose"] = dose_ref.doseArr[2]
				dose["freq"] = "q12"
			elif pt.crcl[-1] >= 20:
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
	vancoClearance = pt.crcl[-1] * 0.06
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

class Case:
	def __init__(self, **kwargs):
		if "cid" not in kwargs.keys():
			self.age = CaseAttr(None, 'age', kwargs['age'])
			self.is_female = CaseAttr(None, 'is_female', kwargs['is_female'])
			self.height = CaseAttr(None, 'height', kwargs['height']) #inches
			
			self.bl_creatinine = CaseAttr(None, 'bl_creatinine', kwargs['bl_creatinine'])
			self.crrt = CaseAttr(None, 'crrt', 1) if 'crrt' in kwargs['comorbid'] else CaseAttr(None, 'crrt', 0)
			self.chf = CaseAttr(None, 'chf', 1) if 'chf' in kwargs['comorbid'] else CaseAttr(None, 'chf', 0)
			self.esld = CaseAttr(None, 'esld', 1) if 'esld' in kwargs['comorbid'] else CaseAttr(None, 'esld', 0)
			self.hd = CaseAttr(None, 'hd', 1) if 'hd' in kwargs['comorbid'] else CaseAttr(None, 'hd', 0)
			self.dm = CaseAttr(None, 'dm', 1) if 'dm' in kwargs['comorbid'] else CaseAttr(None, 'dm', 0)

			wt_obj = Weight(weight=kwargs['weight'])
			wt_obj.dosingWeight = self.setDosingWeight(wt_obj)

			self.wt_arr = [wt_obj]

			cr_obj = Creatinine(creatinine=kwargs['creatinine'])
			self.setCrCl(cr_obj, wt_obj)
			self.cr_arr = [cr_obj]
			self.cid = None
		else:
			cursor = kwargs['cursor']
			cid = kwargs['cid']
			query = "select eid, type, value from vancocalc.case_profile where active=1 and cid=%s"
			case_df = pg.return_df_from_query(cursor, query, (cid,), ["eid", "type", "value"])
			self.cid = cid
			self.age = CaseAttr(case_df[case_df['type'] == 'age']['eid'].item(), 'age', case_df[case_df['type'] == 'age']['value'].item())
			self.is_female = CaseAttr(case_df[case_df['type'] == 'is_female']['eid'].item(), 'is_female', case_df[case_df['type'] == 'is_female']['value'].item())
			self.height = CaseAttr(case_df[case_df['type'] == 'height_in']['eid'].item(), 'height', case_df[case_df['type'] == 'height_in']['value'].item())
			self.bl_creatinine = CaseAttr(case_df[case_df['type'] == 'bl_creatinine']['eid'].item(), 'bl_creatinine', case_df[case_df['type'] == 'bl_creatinine']['value'].item())
			self.crrt = CaseAttr(case_df[case_df['type'] == 'crrt']['eid'].item(), 'crrt', case_df[case_df['type'] == 'crrt']['value'].item())
			self.chf = CaseAttr(case_df[case_df['type'] == 'chf']['eid'].item(), 'chf', case_df[case_df['type'] == 'chf']['value'].item())
			self.esld = CaseAttr(case_df[case_df['type'] == 'esld']['eid'].item(), 'esld', case_df[case_df['type'] == 'esld']['value'].item())
			self.hd = CaseAttr(case_df[case_df['type'] == 'hd']['eid'].item(), 'hd', case_df[case_df['type'] == 'hd']['value'].item())
			self.dm = CaseAttr(case_df[case_df['type'] == 'dm']['eid'].item(), 'dm', case_df[case_df['type'] == 'dm']['value'].item())

			query = "select wtid, weight, dosingWeight, effectivetime from vancocalc.weight where active=1 and cid=%s order by effectivetime asc"
			wt_df = pg.return_df_from_query(cursor, query, (cid,), ["wtid", "weight", "dosingWeight", "effectivetime"])
			self.wt_arr = []
			for index,item in wt_df.iterrows():
				wt_obj = Weight(wtid=item["wtid"], cid=cid, weight=item["weight"], dosingWeight=item["dosingWeight"], effectivetime=item["effectivetime"])
				self.wt_arr.append(wt_obj)

			query = "select crid, creatinine, crcl, effectivetime from vancocalc.creatinine where active=1 and cid=%s order by effectivetime asc"
			cr_df = pg.return_df_from_query(cursor, query, (cid,), ["crid", "creatinine", "crcl", "effectivetime"])
		
			self.cr_arr = []
			for index,item in cr_df.iterrows():
				cr_obj = Creatinine(cid=cid, crid=item['crid'], creatinine=item['creatinine'], crcl=item['crcl'], effectivetime=item['effectivetime'])
				self.cr_arr.append(cr_obj)

	def returnIdealBodyWeight(self, is_female, height):
		if not is_female:
			return (50+(2.3*(height.value-60)))
		else:
			return (45.5 + (2.3 * (height.value-60)))

	def returnAdjustedWeightForIdealBodyWeight(self, idealBodyWeight, actualWeight):
		return (idealBodyWeight + (0.4 * (actualWeight - idealBodyWeight)))

	def setDosingWeight(self, wt_obj):
		idealBodyWeight = self.returnIdealBodyWeight(self.is_female, self.height)

		if (wt_obj.weight > (1.2*idealBodyWeight)):
			return self.returnAdjustedWeightForIdealBodyWeight(idealBodyWeight, wt_obj.weight)
		elif (wt_obj.weight < idealBodyWeight):
			return wt_obj.weight
		else:
			return wt_obj.weight

	def setCrCl(self, cr_obj, wt_obj):

		dosingCreatinine = None
		if ((self.age.value > 65) and (cr_obj.creatinine < 1)):
			dosingCreatinine = 1
		else:
			dosingCreatinine = cr_obj.creatinine
		print(type(wt_obj.dosingWeight))
		print(type(self.age.value))
		if not self.is_female:
			cr_obj.crcl = round(((140-self.age.value)*wt_obj.dosingWeight)/(72*dosingCreatinine), 2)
		else:
			cr_obj.crcl = round(((140-self.age.value)*wt_obj.dosingWeight*0.85)/(72*dosingCreatinine), 2)

	def getCrDict(self):
		r = []
		for item in reversed(self.cr_arr):
			t = {'crid' : item.crid, 'creatinine' : item.creatinine, 'crcl' : item.crcl, 'effectivetime' : item.effectivetime}
			r.append(t)
		return r

	def getWeightDict(self):
		r = []
		for item in reversed(self.wt_arr):
			t = {'wtid' : item.wtid, 'weight' : item.weight, 'dosingWeight' : item.dosingWeight, 'effectivetime' : item.effectivetime}
			r.append(t)
		return r

	def addCr(self, creatinine):
		wt_obj = self.wt_arr[-1]
		cr_obj = Creatinine(creatinine=float(creatinine))
		self.setCrCl(cr_obj, wt_obj)
		self.cr_arr.append(cr_obj)

	def save(self, cid, cursor):
		if self.cid is None:
			insert_q = """
				set schema 'vancocalc'; INSERT INTO case_profile (eid, cid, type, value, active, effectivetime) \
				VALUES
					(public.uuid_generate_v4(), %s, %s, %s, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, 1, now());
			"""
			cursor.execute(insert_q, (cid, 'age', self.age.value, \
				cid, 'is_female', self.is_female.value, \
				cid, 'height_in', self.height.value, \
				cid, 'bl_creatinine', self.bl_creatinine.value, \
				cid, 'chf', self.chf.value, \
				cid, 'esld', self.esld.value, \
				cid, 'dm', self.dm.value, \
				cid, 'crrt', self.crrt.value, \
				cid, 'hd', self.hd.value))
			cursor.connection.commit()

		#if cid is none then new entity
		for i in self.cr_arr:
			if i.cid is None:
				i.cid = cid
				i.save(cid, cursor)

		#if cid is none then new entity
		for j in self.wt_arr:
			if j.cid is None:
				j.cid = cid
				j.save(cid, cursor)

	# def insert_new_vc_case(self, uid, casename, cursor):
	# insert_query = """
	# 		set schema 'vancocalc';
	# 		CREATE EXTENSION IF NOT EXISTS "uuid-ossp" ;

	# 		INSERT INTO cases (cid, uid, casename, active, effectivetime)
	# 		VALUES (public.uuid_generate_v4(), %s, %s, 1, now());
	# """
	# cursor.execute(insert_query, (uid,casename))
	# cursor.connection.commit()
	# return True

class CaseAttr:
	def __init__(self, eid, attr_type, value):
		self.eid = eid
		self.type = attr_type
		self.value = float(value)

class Creatinine:
	def __init__(self, **kwargs):
		if "crid" not in kwargs.keys():
			self.creatinine = float(kwargs['creatinine'])
			self.crcl = None
			self.crid = None
			self.cid = None
			self.effectivetime = None
		else:
			self.creatinine = float(kwargs['creatinine'])
			self.crcl = float(kwargs['crcl'])
			self.crid = kwargs['crid']
			self.cid = kwargs['cid']
			self.effectivetime = kwargs['effectivetime']

	def save(self, cid, cursor):
		if self.crid is None:
			query = "set schema 'vancocalc'; INSERT INTO creatinine (crid, cid, creatinine, crcl, active, effectivetime) \
				VALUES (public.uuid_generate_v4(), %s, %s, %s, 1, now()) RETURNING crid, effectivetime;"
			cursor.execute(query, (cid, self.creatinine, self.crcl))
			all_ids = cursor.fetchall()
			crid = all_ids[0][0]
			effectivetime = all_ids[0][1]
			cursor.connection.commit()
			self.crid = crid
			self.cid = cid
			self.effectivetime = effectivetime

	def delete(self):
		query = "set schema 'vancocalc'; delete from creatinine where crid=%s"
		cursor.execute(query, (self.crid,))
		cursor.connection.commit()

class Weight:
	def __init__(self, **kwargs):
		if "cid" not in kwargs.keys():
			self.weight = float(kwargs['weight'])
			self.dosingWeight = None
			self.wtid = None
			self.cid = None
			self.effectivetime = None
		else:
			self.weight = float(kwargs['weight'])
			self.dosingWeight = float(kwargs['dosingWeight'])
			self.wtid = kwargs['wtid']
			self.cid = kwargs['cid']
			self.effectivetime = kwargs['effectivetime']


	def save(self, cid, cursor):
		if self.wtid is None:
			query = "set schema 'vancocalc'; INSERT INTO weight (wtid, cid, weight, dosingWeight, active, effectivetime) \
				VALUES (public.uuid_generate_v4(), %s, %s, %s, 1, now()) RETURNING wtid, effectivetime;"
			cursor.execute(query, (cid, self.weight, self.dosingWeight))
			all_ids = cursor.fetchall()
			wtid = all_ids[0][0]
			effectivetime = all_ids[0][1]
			cursor.connection.commit()
			self.wtid = wtid
			self.cid = cid
			self.effectivetime = effectivetime

	def delete(self):
		query = "set schema 'vancocalc'; delete from weight where wtid=%s"
		cursor.execute(query, (self.wtid,))
		cursor.connection.commit()

class Dose:
	def __init__(self, troughTarget):
		self.troughTarget = troughTarget

		if troughTarget == 0:
			self.doseArr = [15,12,12,12,12,7,10]
		elif troughTarget == 1:
			self.doseArr = [18,15,15,15,0,7,15]

		self.freqArr = ["8", "12", "12", "24", "0", "post-HD", "24"]


