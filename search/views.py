from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
import snomed_annotator2 as ann2
import utilities.pglib as pg
from nltk.stem.wordnet import WordNetLemmatizer
import utilities.utils as u
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
# import psql_files.annotation2.lemmatizer as lem
# import ml2 as m
# from keras.models import load_model
# Create your views here.

INDEX_NAME='pubmedx1.6'


lowDoseArr = [15,12,12,12,12,7,10]
highDoseArr = [18,15,15,15,0,7,15]
freqArr = ["8", "12", "12", "24", "0", "post-HD", "24"]

def home_page(request):
	return render(request, 'search/home_page.html')


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
	# u.pprint(input_sentence)
	
	# u.pprint(sentence_tuple)
	# print(len(sentence_tuple))

	# condition_ind = int(input("Enter condition index (start or end): "))
	# print(sentence_tuple[condition_ind])

	# condition_id = sentence_tuple[condition_ind][1]

	# print(condition_id)
	
	# sentence_df = pd.DataFrame([[term, sentence_tuple]], columns=['sentence', 'sentence_tuples'])


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
		
	# 
	# conceptid = request.POST['conceptid']
	# description_id = request.POST['description_id']
	# description = request.POST['description_text']
	# if request.POST['action_type'] == 'add_description':
	# 	u.add_description(conceptid, description, cursor)
	# elif request.POST['action_type'] == 'deactivate_description_id':
	# 	u.deactivate_description_id(description_id, cursor)
	# elif request.POST['action_type'] == 'activate_description_id':
	# 	u.activate_description_id(description_id, cursor)
	# elif request.POST['action_type'] == 'add_concept':
	# 	u.add_concept(description, cursor)
	# elif request.POST['action_type'] == 'is_acronym':
	# 	u.acronym_override(description_id, 1, cursor)
	# elif request.POST['action_type'] == 'is_not_acronym':
	# 	u.acronym_override(description_id, 0, cursor)

	cursor.close()
	conn.close()

	return render(request, 'search/concept_override.html', {'payload' : payload_dict, 'acid' : acid, 'adid' : adid})
	# return render(request, 'search/concept_search_home_page.html', 
	# 		{'sr_payload' : None, 'query' : '', 'query_annotation' : None, 'unmatched_terms' : None, 'concepts' : None, 'primary_cids' : None,
	# 			'journals': None, 'start_year' : '', 'end_year' : '', 'at_a_glance' : {'related' : None}, \
	# 			'treatment' : None, 'diagnostic' : None, 'cause' : None, 'condition' : None, \
	# 			'calcs' : None}
	# 		)
	# return HttpResponseRedirect(reverse('search:concept_override', kwargs=payload_dict))

############################## VC LOADING DOSE ##############################
def loading_form(request, cid):
	conn,cursor = pg.return_postgres_cursor()
	case_payload = {}
	if request.method == "GET":
		case = Case(cid=cid, cursor=cursor)
		case_payload['weight'] = case.wt_arr[-1].weight
		case_payload['casename'] = get_casename(cid, cursor)
		case_payload['cid'] = cid
		return render(request, 'vc/loading_form.html', {'case_payload' : case_payload})
	elif request.method == "POST":
		new_wt = request.POST['wt']
		case = Case(cid=cid, cursor=cursor)
		case.addWt(new_wt)
		case.save(cid, cursor)
		return HttpResponseRedirect(reverse('search:loading_rec', args={'cid' : cid}))

def loading_rec(request, cid):
	conn,cursor = pg.return_postgres_cursor()
	if request.method == "GET":
		case_payload = {}
		case = Case(cid=cid, cursor=cursor)
		case_payload['d_obj'] = {'dose' : returnLoadingDoseForPatient(case), 'type' : 'loading', 'freq' : "Loading", 'alert' : None}
		case_payload['cid'] = cid
		case_payload['casename'] = get_casename(cid, cursor)
		cursor.close()
		conn.close()
		return render(request, 'vc/rec_dose.html', {'case_payload' : case_payload})
	elif request.method == "POST":
		d_obj = Dose(dose=request.POST['dose'], freqIndex=None, freqString=request.POST['freq'], alert=request.POST['alert'])
		case = Case(cid=cid, cursor=cursor)
		case.addDose(d_obj)
		case.save(cid, cursor)
		cursor.close()
		conn.close()
		return HttpResponseRedirect(reverse('search:vc_case_view', args=(cid,)))

############################## VC MAINTENANCE DOSE ##########################

def maintenance_form(request, cid):
	conn, cursor = pg.return_postgres_cursor()
	case_payload = {}
	if request.method == "GET":
		case = Case(cid=cid, cursor=cursor)
		case_payload['casename'] = get_casename(cid, cursor)
		case_payload['cid'] = cid
		case_payload['weight'] = case.wt_arr[-1].weight
		cursor.close()
		conn.close()
		return render(request, 'vc/maintenance_form.html', {'case_payload' : case_payload})
	elif request.method == "POST":
		case = Case(cid=cid, cursor=cursor)
		cr = request.POST['cr']
		weight = request.POST['wt']
		case = Case(cid=cid, cursor=cursor)
		case.addCr(cr)
		case.addWt(weight)
		case.save(cid, cursor)
		cursor.close()
		conn.close()
		return HttpResponseRedirect(reverse('search:maintenance_rec', kwargs={'cid' : cid}))

def maintenance_rec(request, cid):
	conn,cursor = pg.return_postgres_cursor()
	case_payload = {}
	if request.method == "GET":
		case = Case(cid=cid, cursor=cursor)
		d_obj = returnInitialMaintenanceDose(case)
		
		if d_obj.dose == 0 and d_obj.freqString != None:
			d_obj.alert = "Discuss with pharmacy"
		elif case.returnARF() and case.hd.value != 1 and case.crrt.value != 1:
			d_obj.alert = "Patient may be in acute renal failure. Trough should be drawn before the 3rd dose"
		elif d_obj.freqString == "q8":
			d_obj.alert = "Trough should be drawn before the 5th dose. Consider checking BMPs twice daily."
		else:
			d_obj.alert = "Trough should be drawn before the 4th dose."
		# case.addDose(d_obj) Doesn't seem like this should be added yet

		case_payload['casename'] = get_casename(cid, cursor)
		case_payload['cid'] = cid
		case_payload['d_obj'] = {'dose' : d_obj.dose, 'freq' : d_obj.freqString, 'type' : 'maintenance', 'alert' : d_obj.alert}
		cursor.close()
		conn.close()
		return render(request, 'vc/rec_dose.html', {'case_payload' : case_payload})
	else:
		d_obj = Dose(dose=request.POST['dose'], freqIndex=None, freqString=request.POST['freq'], alert=request.POST['alert'])
		case = Case(cid=cid, cursor=cursor)
		case.addDose(d_obj)
		case.save(cid, cursor)
		cursor.close()
		conn.close()
		return HttpResponseRedirect(reverse('search:vc_case_view', args=(cid,)))

############################## VC CUSTOM DOSE ##############################

def custom_dose_form(request, cid):
	conn, cursor = pg.return_postgres_cursor()
	if request.method == "GET":
		case_payload = {}
		case_payload['casename'] = get_casename(cid, cursor)

		case_payload['rec_dose'] = {'dose' : request.GET['dose'], 'freq':request.GET['freq'], 'alert' : request.GET['alert'] }
		cursor.close()
		conn.close()
		return render(request, 'vc/custom_dose.html', {'case_payload' : case_payload})
	elif request.method == "POST":
		dose = int(request.POST['doseSelect'])
		freqString = str(request.POST['freqSelect'])
		alert = str(request.POST['alert'])
		d_obj = Dose(dose=dose, freqIndex=None, freqString=freqString, alert=alert)
		case = Case(cid=cid, cursor=cursor)
		case.addDose(d_obj)
		case.save(cid, cursor)
		cursor.close()
		conn.close()
		return HttpResponseRedirect(reverse('search:vc_case_view', args=(cid,)))

############################## VC REDOSE DOSE ###############################

def redose_form(request, cid):
	conn, cursor = pg.return_postgres_cursor()
	case_payload = {}

	if request.method == "GET":
		case = Case(cid=cid, cursor=cursor)
		case_payload['casename'] = get_casename(cid, cursor)
		case_payload['cid'] = cid
		case_payload['weight'] = case.wt_arr[-1].weight
		cursor.close()
		conn.close()
		return render(request, 'vc/redose_form.html', {'case_payload' : case_payload})
	elif request.method == "POST":
		weight=request.POST['wt']
		cr = request.POST['cr']
		trough = float(request.POST['trough'])
		doseNum = int(request.POST['doseNum'])
		case = Case(cid=cid, cursor=cursor)
		case.addCr(cr)
		case.addWt(weight)
		case.addTrough(trough, doseNum)
		case.save(cid, cursor)
		cursor.close()
		conn.close()
		return HttpResponseRedirect(reverse('search:redose_rec', args=(cid,)))

def redose_rec(request, cid):
	conn, cursor = pg.return_postgres_cursor()
	case_payload = {}

	if request.method == "GET":
		case = Case(cid=cid, cursor=cursor)
		d_obj = returnNewMaintenanceDoseForPatient(case)
		case_payload['casename'] = get_casename(cid, cursor)
		case_payload['cid'] = cid
		case_payload['d_obj'] = {'dose' : d_obj.dose, 'freq' : d_obj.freqString, 'type' : 'maintenance', 'alert' : d_obj.alert}
		return render(request, 'vc/rec_dose.html', {'case_payload' : case_payload})
	elif request.method == "POST":
		case = Case(cid=cid, cursor=cursor)
		d_obj = returnNewMaintenanceDoseForPatient(case)
		case.addDose(d_obj)
		case.save(cid, cursor)
		return HttpResponseRedirect(reverse('search:vc_case_view', args=(cid,)))
		


############################## VC CASE / USER MGMT ##########################

def vc_main(request):
	vc_payload = {}
	conn, cursor = pg.return_postgres_cursor()
	if request.method == "POST":
		username=request.POST['username']
		u.insert_new_vc_user(username, cursor)
	
	user_df = get_vc_users(cursor)
	user_payload = get_user_payload(user_df)

	return render(request, 'vc/main.html', {'user_payload' : user_payload})

def vc_cases(request, uid):
	cases_payload = {}
	conn, cursor = pg.return_postgres_cursor()
	username = ""
	cases_df = None

	if request.method == "POST":

		username = get_username(uid, cursor)
		casename = request.POST['casename']
		u.insert_new_vc_case(uid, casename, cursor)
	else:
		username = get_username(uid, cursor)
	
	cases_df = get_vc_cases(cursor, uid)
	cases_payload['cases'] = get_cases_payload(cases_df)
	cases_payload['uid'] = uid
	cases_payload['username'] = username

	return render(request, 'vc/cases.html', {'cases_payload' : cases_payload})

def vc_new_case(request, uid):
	new_case_payload = {}
	new_case_payload['uid'] = uid
	new_case_payload['username'] = request.GET['username']
	return render(request, 'vc/new_case.html', {'new_case_payload' : new_case_payload})

def vc_case_view(request, cid):
	case_payload = {}
	conn, cursor = pg.return_postgres_cursor()
	casename=""
	username=""
	case = None
	uid = ""

	if request.method == "GET":
		casename = get_casename(cid, cursor)
		case = Case(cid=cid, cursor=cursor)
	elif 'create' in request.POST:
		conn, cursor = pg.return_postgres_cursor()
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
		args = {'age' : age, 'is_female' : is_female, 'height' : height, 'weight' : weight, \
			'creatinine' : creatinine, 'bl_creatinine' : bl_creatinine, 'comorbid' : comorbid, \
			'targetTrough' : targetTrough, 'indication' : indication}
		case = Case(**args)
		case.save(cid, cursor)

	elif request.method=='POST':
		conn, cursor = pg.return_postgres_cursor()
		creatinine = request.POST['cr']
		case = Case(cid=cid, cursor=cursor)
		case.addCr(creatinine)
		case.save(cid, cursor)

	case_payload['uid'] = uid
	case_payload['casename'] = casename
	case_payload['cid'] = cid
	case_payload = get_case_payload(case, case_payload)

	return render(request, 'vc/case_view.html', {'case_payload' : case_payload})


def get_case_payload(case, case_payload):
	case_payload["chf"] = case.chf.value
	case_payload["dm"] = case.dm.value
	case_payload["esld"] = case.esld.value
	case_payload["crrt"] = case.crrt.value
	case_payload["hd"] = case.hd.value
	case_payload["crDose"] = case.getCrDoseDict()
	case_payload["weight"] = case.getWeightDict()

	return case_payload


def get_username(uid, cursor):
	query = "select username from vancocalc.users where uid=%s"
	user_df = pg.return_df_from_query(cursor, query, (uid,), ["username"])
	
	return user_df["username"].item()

def get_casename(cid, cursor):
	query = "select casename from vancocalc.cases where cid=%s"
	user_df = pg.return_df_from_query(cursor, query, (cid,), ["casename"])
	
	return user_df["casename"].item()

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


def returnLoadingDoseForPatient(case):
	if (case.age.value >= 70):
		dose = 15*case.wt_arr[-1].weight
	else:
		dose = 20*case.wt_arr[-1].weight

	if (dose > 2000):
		dose = 2000
	return returnRoundedDose(dose)

def returnInitialMaintenanceDose(case):
	d = None
	dfi = None
	dfs = None

	if case.targetTrough == 0:
		if case.crrt.value == 1:
			d = lowDoseArr[6]
			dfi = 1
			dfs = "q12"
		elif case.hd.value == 1:
			d = lowDoseArr[5]
			dfi = 5
			dfs = "post-HD"
		else:
			if case.cr_arr[-1].crcl >= 70:
				if ((case.age.value > 50) or (case.esld.value == 1) or (case.chf.value == 1) or (case.dm.value == 1)):
					d = lowDoseArr[0]
					dfi = 1
					dfs = "q12"
				else:
					d = lowDoseArr[1]
					dfi = 0
					dfs = "q8"
			elif (case.cr_arr[-1].crcl >= 40):
				d = lowDoseArr[2]
				dfi = 1
				dfs = "q12"
			elif (case.cr_arr[-1].crcl >= 20):
				d = lowDoseArr[3]
				dfi = 3
				dfs = "q24"
			else:
				d = lowDoseArr[4]
				dfi = 6
				dfs = "q48"
	else: #trough target is 15-20
		if case.crrt.value == 1:
			d = highDoseArr[6]
			dfi = 3 
			dfs = "q24"
		elif case.hd.value == 1:
			d = highDoseArr[5]
			dfi = 5
			dfs = "post-HD"
		else:
			if case.cr_arr[-1].crcl >= 70:
				if ((case.age.value > 50) or (case.esld.value == 1) or (case.chf.value == 1) or (case.dm.value == 1)):
					d = highDoseArr[0]
					dfi = 1
					dfs = "q12"
				else:
					d = highDoseArr[1]
					dfi = 0
					dfs = "q8"
			elif case.cr_arr[-1].crcl >= 40:
				d = highDoseArr[2]
				dfi = 1
				dfs = "q12"
			elif case.cr_arr[-1].crcl >= 20:
				d = highDoseArr[3]
				dfi = 3
				dfs = "q24"
			else:
				d = highDoseArr[4]
				dfi = 7
				dfs = "single loading dose followed by kintetics"
	d = d * case.wt_arr[-1].dosingWeight
	d = returnRoundedDose(d)

	d_obj = Dose(dose=d, freqIndex=dfi, freqString=dfs, alert=None)

	return d_obj

def returnNewMaintenanceDoseForPatient(case):
	vd = 0.7 * case.wt_arr[-1].dosingWeight
	vancoClearance = case.cr_arr[-1].crcl * 0.06
	ke = vancoClearance / vd
	halfLife = 0.693/ke
	
	arf = case.returnARF()

	dose = None
	priorDose = case.dose_arr[-1]

	lowTrough = None
	highTrough = None
	dialysisMultiplier = None
	trough_obj = case.trough_arr[-1]
	trough = trough_obj.trough
	if trough_obj.beforeDoseNum == 3:
		trough = trough*1.3

	if (case.targetTrough == 0):
		lowTrough = 10.0
		highTrough = 15.0
		dialysisMultiplier = 12
	else:
		lowTrough = 15.0
		highTrough = 20.0
		dialysisMultiplier = 15

	if (case.hd.value == 1 or case.crrt.value == 1):
		newDose = priorDose.dose * (dialysisMultiplier / trough)
		newDose = returnRoundedDose(newDose)
		if case.hd.value == 1:
			return Dose(dose=newDose, freqIndex=5, freqString="post-HD", alert=None)
		else:
			return Dose(dose=newDose, freqIndex=3, freqString="q24", alert=None)
	# Supratherapeutic by a lot
	elif (trough > (highTrough + 2)):
		return Dose(dose=0, freqIndex=None, freqString=None, alert="Consider holding a dose or re-checking trough daily before resuming vancomycin. Discuss with pharmacy.")
	elif arf:
		arfMultiplier = None
		if (case.targetTrough == 1):
			arfMultiplier = 12
		else:
			arfMultiplier = 17

		new_dose = (arfMultiplier * vd * (1-(math.exp(-ke*halfLife)))) / (math.exp(-ke*halfLife))
		new_dose = returnRoundedDose(new_dose)
		freq = returnRoundedFrequency(halfLife*1.5)
		freqString = "q" + str(freq)
		if (trough > highTrough):
			if new_dose > priorDose.dose:
				return Dose(dose=priorDose.dose, freqIndex=None, freqString=priorDose.freqString, alert="Patient in acute renal failure. Monitor kidney function and urine output closely on current dose.")
			else:
				return Dose(dose=new_dose, freqIndex=None, freqString=freqString, alert="Due to a supratherapeutic trough, consider holding next dose before resuming recommended dosing schedule. Dose adjusted for acute renal failure.")
		else:
			if (trough > lowTrough and new_dose > priorDose.dose):
				return Dose(dose=priorDose.dose, freqIndex=None, freqString=priorDose.freqString, alert="Patient in acute renal failure. Monitor kidney function and urine output closely on current dose")
			else:
				return Dose(dose=new_dose, freqIndex=None, freqString=freqString, alert="Dose adjusted for acute renal failure")
	elif ((trough >= lowTrough) and (trough <= highTrough) and not arf):
		return Dose(dose=priorDose.dose, freqIndex=priorDose.freqIndex, freqString=priorDose.freqString, alert=priorDose.alert)
	elif (trough > highTrough):
		if (priorDose.dose >= 500):
			new_dose = priorDose.dose -250
			return Dose(dose=new_dose, freqIndex=priorDose.freqIndex, freqString=priorDose.freqString, alert="Due to a supratherapeutic trough, consider holding next dose before resuming recommended dosing schedule")
		else:
			return Dose(dose=0, freqIndex=None, freqString=None, alert="Consider holding a dose or re-checking trough before resuming vancomycin. Recommend discussing with pharmacy to re-dose when trough enters normal range")
	# If much less than trough goal
	elif (trough < lowTrough -5):
		new_dose = returnRoundedDose(priorDose.dose * 1.75)
		return Dose(dose=new_dose, freqIndex=priorDose.freqIndex, freqString=priorDose.freqString, alert=None)
	#If within 2, go ahead and use prior dose
	elif (trough > lowTrough-2):
		return Dose(dose=priorDose.dose, freqIndex=priorDose.freqIndex, freqString=priorDose.freqString, alert=None)
	else:
		new_dose = priorDose.dose + 250
		return Dose(dose=new_dose, freqIndex=priorDose.freqIndex, freqString=priorDose.freqString, alert=None)

def returnRoundedFrequency(freq):
	diff8 = abs(freq-8)
	diff12 = abs(freq-12)
	diff24 = abs(freq-24)
	diff48 = abs(freq-48)

	if (diff8 < diff12):
		return 8
	elif (diff12 < diff24):
		return 12
	elif (diff24 < diff48):
		return 24
	else:
		return 48

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

def getJournals(request):
	conn,cursor = pg.return_postgres_cursor()
	query = "select iso_abbrev from pubmed.journals"
	journals = pg.return_df_from_query(cursor, query, (None,), ['journals'])
	data = {"journals" : journals['journals'].tolist()}
	
	return JsonResponse(data)


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
	unmatched_terms = None
	query_type = ''
	primary_cids = None
	query = ''
	filters = {}
	if request.method == 'GET': 
		parsed = urlparse.urlparse(request.path)
		parsed = parse_qs(parsed.path)
		# query = parse_qs(parsed.path)['/search/query'][0]
		# return render(request, 'search/concept_search_home_page.html', 
		# 	{'sr_payload' : None, 'query' : '', 'concepts' : None, 'query_annoation' : None, 'unmatched_terms' : None,
		# 		'journals': None, 'start_year' : '', 'end_year' : '', 'at_a_glance' : {'related' : None}, \
		# 		'treatment' : None, 'diagnostic' : None, 'cause' : None, 'condition' : None, \
		# 		'calcs' : None}
		# 	)
		query = parsed['/search/query'][0]

		if 'query_annotation[]' in parsed:
			primary_cids = parsed['query_annotation[]']
		
		if unmatched_terms in parsed:
			unmatched_terms = parsed['unmatched_terms'][0]
		
		if 'pivot_cid' in parsed:
			primary_cids.append(['pivot_cid'])

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

	elif request.method == 'POST':
		data = json.loads(request.body)
		query = data['query']
		filters = get_query_filters(data)
		query_type = data['query_type']
		if 'query_annotation' in data:
			primary_cids = data['query_annotation']

		if 'pivot_cid' in data:
			primary_cids.append(data['pivot_cid'])

		# html = render(request, 'search/concept_search_results_page.html', {'sr_payload' : sr_payload, 'query' : query, 'query_annotation' : query_annotation, \
		# 		'unmatched_terms' : unmatched_terms, 'concepts' : query_concepts_dict, \
		# 		'journals': filters['journals'], 'start_year' : filters['start_year'], 'end_year' : filters['end_year'], \
		# 		'treatment' : treatment_dict, 'diagnostic' : diagnostic_dict, 'cause' : cause_dict, 'condition' : condition_dict, \
		# 		'calcs' : calcs_json})
	sr = dict()
	query_concepts_dict = dict()
	related_dict = {}
	treatment_dict = {}
	condition_dict = {}
	diagnostic_dict = {}
	cause_dict = {}
	calcs_json = {}
	unmatched_terms = 'None'
	flattened_query = None

	if query_type == 'pivot':
		query_concepts_df = pd.DataFrame(primary_cids, columns=['acid'])
		full_query_concepts_list = ann2.query_expansion(query_concepts_df['acid'], cursor)

		unmatched_terms = data['unmatched_terms']

			
		pivot_term = data['pivot_term']
		query = data['query'] + ' ' + pivot_term
		params = filters


		flattened_query_concepts_list = get_flattened_query_concept_list(query_concepts_df)
		query_concepts_dict = get_query_arr_dict(full_query_concepts_list)

		es_query = {"from" : 0, \
				 "size" : 500, \
				 "query": get_query(full_query_concepts_list, unmatched_terms, filters['journals'], filters['start_year'], filters['end_year'], \
				 	["title_conceptids^5", "abstract_conceptids.*"], cursor)}

		sr = es.search(index=INDEX_NAME, body=es_query, request_timeout=100000)

		sr_payload = get_sr_payload(sr['hits']['hits'])

		treatment_dict, diagnostic_dict, condition_dict, cause_dict = get_related_conceptids(full_query_concepts_list, primary_cids,
					unmatched_terms, filters, cursor)

		

	elif query_type == 'keyword':

		query = ann2.clean_text(query)

		original_query_concepts_list = []
		if query.upper() != query:
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
						 "size" : 300, \
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
		print("GET")
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
	concepts = query_concepts_df['acid'].tolist()

	query = "select distinct on (title, t1.desc, url) title, t1.desc, url from annotation2.mdc_staging t1 where acid in %s"
	calcs = pg.return_df_from_query(cursor, query, (tuple(concepts),), ['title', 'desc', 'url'])

	calc_json = []
	for ind,item in calcs.iterrows():
		calc_json.append({'title' : item['title'], 'desc' : item['desc'], 'url' : item['url']})

	return calc_json

def rollups(cids_df, cursor):
	if len(cids_df.index) > 0:
		params = (tuple(cids_df['acid']), tuple(cids_df['acid']))

		query = """ 
			select 
				child_acid
				,parent_acid 
			from snomed2.transitive_closure_acid 
			where child_acid in %s and parent_acid in %s 
				and parent_acid not in (select parent_acid from snomed2.transitive_closure_acid group by parent_acid having count(*) > 1000)

			"""

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

	# cursor.connection.commit()



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
	unmatched_terms = 'None'
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
			{"term": {"article_type" : "News"}}
			]
	return filt

def get_concept_string(conceptid_series):
	result_string = ""
	for item in conceptid_series:
		result_string += item + " "
	
	return result_string.strip()

def get_query(full_conceptid_list, unmatched_terms, journals, start_year, end_year, fields_arr, cursor):
	es_query = {}
	if unmatched_terms == 'None':
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

		tx_df = pg.return_df_from_query(cursor, query, (tuple(query_concept_list[0]),), ["acid"])
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
		### THIS WONT WORK WITH MULTIPLE CONCEPTS

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
		# return pd.DataFrame([], columns=["conceptid", "concept_type"])

def get_related_conceptids(query_concept_list, original_query_concepts_list, unmatched_terms, filters, cursor):
	result_dict = dict()
	es = es_util.get_es_client()

	es_query = get_query(query_concept_list, unmatched_terms, \
						 	filters['journals'], filters['start_year'], filters['end_year']\
						 	,["title_conceptids^5"], cursor)

	scroller = es_util.ElasticScroll(es, es_query)

	title_match_cids_df = pd.DataFrame()
	while scroller.has_next:
		article_list = scroller.next()
		if article_list is not None: 
			title_match_cids_df = title_match_cids_df.append(get_title_cids(article_list), sort=False)
		else:
			break

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

		agg_tx = get_query_concept_types_df_3(title_match_cids_df, query_concept_list, cursor, 'treatment')

		if not agg_tx.empty:
			agg_tx = agg_tx.drop_duplicates(subset=['acid', 'pmid'])
			agg_tx['count'] = 1
			agg_tx = agg_tx.groupby(['acid'], as_index=False)['count'].sum()

			agg_tx = ann2.add_names(agg_tx)
			sub_dict['treatment'] = rollups(agg_tx, cursor)


		agg_diagnostic = get_query_concept_types_df_3(title_match_cids_df, query_concept_list, cursor, 'diagnostic')

		if len(agg_diagnostic) > 0:
			agg_diagnostic = agg_diagnostic.drop_duplicates(subset=['acid', 'pmid'])
			agg_diagnostic['count'] = 1
			agg_diagnostic = agg_diagnostic.groupby(['acid'],  as_index=False)['count'].sum()
			
			agg_diagnostic = ann2.add_names(agg_diagnostic)
			sub_dict['diagnostic'] = rollups(agg_diagnostic, cursor)
			

		agg_cause = get_query_concept_types_df_3(title_match_cids_df, query_concept_list, cursor, 'cause')

		if len(agg_cause) > 0:
			agg_cause = agg_cause.drop_duplicates(subset=['acid', 'pmid'])
			agg_cause['count'] = 1
			agg_cause = agg_cause.groupby(['acid'],  as_index=False)['count'].sum()
			agg_cause = ann2.add_names(agg_cause)
			sub_dict['cause'] = rollups(agg_cause, cursor)


		agg_condition = get_query_concept_types_df_3(title_match_cids_df, query_concept_list, cursor, 'condition')

		if len(agg_condition) > 0:
			agg_condition = agg_condition.drop_duplicates(subset=['acid', 'pmid'])
			agg_condition['count'] = 1
			agg_condition = agg_condition.groupby(['acid'],  as_index=False)['count'].sum()
			agg_condition = ann2.add_names(agg_condition)
			agg_condition = agg_condition.sort_values(['count'], ascending=False)
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

				# u.pprint(t['conceptid'])
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
			self.targetTrough = kwargs['targetTrough']
			self.indication = kwargs['indication']
			self.dose_arr = []
			self.trough_arr = []
		else:
			cursor = kwargs['cursor']
			cid = kwargs['cid']
			query = "select eid, type, value, str_value from vancocalc.case_profile where active=1 and cid=%s"
			case_df = pg.return_df_from_query(cursor, query, (cid,), ["eid", "type", "value", "str_value"])
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
			self.targetTrough = case_df[case_df['type'] == 'targetTrough']['value'].item()
			self.indication = case_df[case_df['type'] == 'indication']['str_value'].item()

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

			query = "select did, dose, freqIndex, freqString, alert, effectivetime from vancocalc.doses where active=1 and cid=%s order by effectivetime asc"
			dose_df = pg.return_df_from_query(cursor, query, (cid,), ["did", "dose", "freqIndex", "freqString", "alert", "effectivetime"])
			self.dose_arr = []
			for index,item in dose_df.iterrows():
				dose_obj = Dose(cid=cid,did=item['did'], dose=item['dose'], freqIndex=item['freqIndex'], freqString=item['freqString'], alert=item['alert'], effectivetime=item['effectivetime'])
				self.dose_arr.append(dose_obj)

			query = "select tid, trough, beforeDoseNum, effectivetime from vancocalc.trough where active=1 and cid=%s order by effectivetime asc"
			trough_df = pg.return_df_from_query(cursor, query, (cid,), ["tid", "trough", "beforeDoseNum", "effectivetime"])
			self.trough_arr = []
			for index,item in trough_df.iterrows():
				trough_obj = Trough(cid=cid, tid=item['tid'], trough=item['trough'], beforeDoseNum=item['beforeDoseNum'], effectivetime=item['effectivetime'])
				self.trough_arr.append(trough_obj)


	def returnARF(self):
		crCount = len(self.cr_arr)
		currCr = self.cr_arr[-1].creatinine

		arfFromBaseline = False
		arfFromPrior = False

		if (self.hd.value != 1):
			if (self.bl_creatinine.value):
				if ((currCr >= 1.5*self.bl_creatinine.value) or (currCr >= (self.bl_creatinine.value + 0.3))):
					arfFromBaseline = True
			
			if (crCount < 3):
				lastCr = self.cr_arr[-2].creatinine

				if ((currCr >= 1.5*lastCr) or (currCr >= (lastCr + 0.3))):
					arfFromPrior = True
			else:
				lastCr = self.cr_arr[-2].creatinine
				secondLastCr = self.cr_arr[-3].creatinine

				if ((currCr >= 1.5*lastCr) or (currCr >= 1.5*secondLastCr) \
					or (currCr >= (lastCr + 0.3)) or (currCr >= (secondLastCr + 0.3))):
					arfFromPrior = True

		if arfFromBaseline or arfFromPrior:
			return True
		else:
			return False

	def addTrough(self, trough, beforeDoseNum):
		t_obj = Trough(trough=trough, beforeDoseNum=beforeDoseNum)
		self.trough_arr.append(t_obj)
	
	def addDose(self, d_obj):
		self.dose_arr.append(d_obj)

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

		if not self.is_female:
			cr_obj.crcl = round(((140-self.age.value)*wt_obj.dosingWeight)/(72*dosingCreatinine), 2)
		else:
			cr_obj.crcl = round(((140-self.age.value)*wt_obj.dosingWeight*0.85)/(72*dosingCreatinine), 2)

	def getCrDoseDict(self):
		res = []
		tmp = []
		tmp.extend(self.cr_arr)
		tmp.extend(self.dose_arr)
		tmp.extend(self.trough_arr)
		tmp = sorted(tmp, key=lambda x: x.effectivetime, reverse=True)

		for c in tmp:
			if type(c).__name__ == 'Creatinine':
				t = {'type' : 'creatinine', 'crid' : c.crid, 'creatinine' : c.creatinine, 'crcl' : c.crcl, 'effectivetime' : c.effectivetime}
				res.append(t)
			elif type(c).__name__ == 'Dose':
				t = {'type' : 'dose', 'did' : c.did, 'dose' : c.dose, 'freqString' : c.freqString, 'alert' : c.alert, 'effectivetime' : c.effectivetime}
				res.append(t)
			elif type(c).__name__ == 'Trough':
				t = {'type' : 'trough', 'tid' : c.tid, 'trough' : c.trough, 'effectivetime' : c.effectivetime}
		return res

		# for cr in reversed(self.cr_arr):
		# 	for d in reversed(self.dose_arr):
		# 		if 
		# 	t = {'crid' : cr.crid, 'creatinine' : cr.creatinine, 'crcl' : cr.crcl, 'effectivetime' : cr.effectivetime}
		# 	r.append(t)
		# return r

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

	def addWt(self, weight):
		wt_obj = Weight(weight=weight)
		wt_obj.dosingWeight = self.setDosingWeight(wt_obj)
		self.wt_arr.append(wt_obj)

	def save(self, cid, cursor):
		if self.cid is None:
			insert_q = """
				set schema 'vancocalc'; INSERT INTO case_profile (eid, cid, type, value, str_value, active, effectivetime) \
				VALUES
					(public.uuid_generate_v4(), %s, %s, %s, NULL, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, NULL, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, NULL, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, NULL, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, NULL, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, NULL, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, NULL, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, NULL, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, NULL, 1, now()),
					(public.uuid_generate_v4(), %s, %s, NULL, %s, 1, now()),
					(public.uuid_generate_v4(), %s, %s, %s, NULL, 1, now());
			"""
			cursor.execute(insert_q, (cid, 'age', self.age.value, \
				cid, 'is_female', self.is_female.value, \
				cid, 'height_in', self.height.value, \
				cid, 'bl_creatinine', self.bl_creatinine.value, \
				cid, 'chf', self.chf.value, \
				cid, 'esld', self.esld.value, \
				cid, 'dm', self.dm.value, \
				cid, 'crrt', self.crrt.value, \
				cid, 'hd', self.hd.value, \
				cid, 'indication', self.indication, \
				cid, 'targetTrough', self.targetTrough))
			cursor.connection.commit()

		#if cid is none then new entity
		for i in self.cr_arr:
			if i.cid is None:
				i.save(cid, cursor)

		#if cid is none then new entity
		for j in self.wt_arr:
			if j.cid is None:
				j.save(cid, cursor)

		for d in self.dose_arr:
			if d.cid is None:
				d.save(cid, cursor)

		for t in self.trough_arr:
			if t.tid is None:
				t.save(cid, cursor)


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
		if "wtid" not in kwargs.keys():
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
	def __init__(self, **kwargs):
		if "did" not in kwargs.keys():
			self.dose = kwargs['dose']
			self.freqIndex = kwargs['freqIndex']
			self.freqString = kwargs['freqString']
			self.alert = kwargs['alert']
			self.did = None
			self.effectivetime = None
			self.cid = None
		else:
			self.dose = kwargs['dose']
			self.freqIndex = kwargs['freqIndex']
			self.freqString = kwargs['freqString']
			self.did = kwargs['did']
			self.effectivetime = kwargs['effectivetime']
			self.cid = kwargs['cid']
			self.alert = kwargs['alert']

	def save(self, cid, cursor):

		if self.did is None:

			query = "set schema 'vancocalc'; INSERT into doses (did, cid, dose, freqIndex, freqString, alert, active, effectivetime) \
				VALUES (public.uuid_generate_v4(), %s, %s, %s, %s, %s, 1, now()) RETURNING did, effectivetime;"
			cursor.execute(query, (cid, self.dose, self.freqIndex, self.freqString, self.alert))
			all_ids = cursor.fetchall()
			did = all_ids[0][0]
			effectivetime = all_ids[0][1]
			cursor.connection.commit()
			self.did = did
			self.effectivetime = effectivetime
			self.cid = cid

class Trough():
	def __init__(self, **kwargs):
		if "tid" not in kwargs.keys():
			self.trough = float(kwargs['trough'])
			self.beforeDoseNum = int(kwargs['beforeDoseNum'])
			self.tid = None
			self.cid = None
			self.effectivetime = None
		else:
			self.trough = float(kwargs['trough'])
			self.beforeDoseNum = int(kwargs['beforeDoseNum'])
			self.tid = kwargs['tid']
			self.cid = kwargs['cid']
			self.effectivetime = kwargs['effectivetime']

	def save(self, cid, cursor):
		if self.tid is None:
			query = "set schema 'vancocalc'; INSERT INTO trough (tid, cid, trough, beforeDoseNum, active, effectivetime) \
				VALUES (public.uuid_generate_v4(), %s, %s, %s, 1, now()) RETURNING tid, effectivetime;"
			cursor.execute(query, (cid, self.trough, self.beforeDoseNum))
			all_ids = cursor.fetchall()
			tid = all_ids[0][0]
			effectivetime = all_ids[0][1]
			cursor.connection.commit()
			self.tid = tid
			self.cid = cid
			self.effectivetime = effectivetime
