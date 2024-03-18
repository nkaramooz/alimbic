# Functions here are used to make changes to the underlying 
# knowledge graph.

from django.shortcuts import render
from django.http import JsonResponse
import utilities.pglib as pg
from django.template.loader import render_to_string


# Main Graph View
def graph(request):
	return render(request, 'graph/graph_home.html')


# Returns information on concept ID (acid) to graph view
def get_acid(request):
	acid = request.POST["acid"]
	div = request.POST["results-div"]
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
		from annotation2.add_adid_acronym t1
		left join annotation2.downstream_root_cid t2
		on t1.acid = t2.acid
		where t1.acid = %s
	"""
	df = pg.return_df_from_query(query, (acid,), \
		["adid", "acid", "cid", "term", "term_lower",'word', "word_ord", "is_acronym"])
	div = request.POST["results-div"]
	return JsonResponse(get_payload(div, df=df))


# Returns information on description ID (adid) to graph view
def get_adid(request):
	adid = request.POST["adid"]
	div = request.POST["results-div"]
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
		from annotation2.add_adid_acronym t1
		left join annotation2.downstream_root_cid t2
		on t1.acid = t2.acid
		where adid = %s
	"""
	df = pg.return_df_from_query(query, (adid,), \
	 	["adid", "acid", "cid", "term", "term_lower",'word', "word_ord", "is_acronym"])
	div = request.POST["results-div"]
	return JsonResponse(get_payload(div, df=df))


# Search concepts by term (lowered)
def get_term(request):
	term = request.POST["term"]
	div = request.POST["results-div"]
	df = pg.get_term(term)
	div = request.POST["results-div"]
	return JsonResponse(get_payload(div, df=df))


# Writes a new condition <> treatment relationship
# provided by the user
# TODO: Change front end values to readable strings and map to integer on backend
def post_label_rel(request):
	condition_acid = request.POST["condition_acid"]
	treatment_acid = request.POST["treatment_acid"]
	rel = request.POST["rel"]
	alert = pg.add_labelled_treatment(condition_acid, treatment_acid, rel)
	alert += " condition_acid : %s " % condition_acid
	alert += ". treatment_acid : %s " % treatment_acid
	alert += ". relationship type: %s " % rel
	div = request.POST["results-div"]
	print(alert)
	return JsonResponse(get_payload(div, alert=alert))


# Gets the parents and children of the concept ID
def get_rel(request):
	payload = {}
	payload["data"] = []
	acid = request.POST["acid"]
	
	# Child block
	df = pg.get_children(acid)
	child_div = request.POST["results-div[children]"]
	child_payload = get_payload(child_div, alert="No children found.")  if len(df.index) == 0 else \
		get_payload(child_div, df=df, header="Children")
	payload["data"].append(child_payload["data"][0])

	# Parent block	
	df = pg.get_parents(acid)
	parent_div = request.POST["results-div[parents]"]
	parent_payload = get_payload(parent_div, alert="No parents found.") if len(df.index) == 0 else \
		get_payload(parent_div, df=df, header="Parents")

	payload["data"].append(parent_payload["data"][0])
	return JsonResponse(payload)


# Creates a concept that will be added to the knowledge graph
# the next time the graph is updated.
def create_concept(request): 
	term = request.POST["term"]
	alert = pg.create_concept(term)
	div = request.POST["results-div"]
	return JsonResponse(get_payload(div, alert=alert))


# Marks a concept to be made inactive the next time the knowledge
# graph is updated.
def deactivate_concept(request):
	acid = request.POST["acid"]
	alert = pg.deactivate_concept(acid)
	return JsonResponse(get_payload(div, alert=alert))


# Adds a new description for the conceptid.
def new_description(request):
	acid = request.POST["acid"]
	term = request.POST["term"]
	alert = pg.add_description(acid, term)
	div = request.POST["results-div"]
	return JsonResponse(get_payload(div, alert=alert))


# Marks a description ID (ADID) to be made inactive the next time
# the knowledge graph is updated.
def deactivate_description(request):
	adid = request.POST["adid"]
	alert = pg.deactivate_description(adid)
	div = request.POST["results-div"]
	return JsonResponse(get_payload(div, alert=alert))


# Add or remove a parent/child relationship
def modify_parent(request):
	child_acid = request.POST["child-acid"]
	parent_acid = request.POST["parent-acid"]

	# 0 = Delete, 1 = Add
	add = request.POST["update-rel"]
	alert = pg.modify_relationship(child_acid, parent_acid, add)
	div = request.POST["results-div"]
	return JsonResponse(get_payload(div, alert=alert))


# Set the acronym flag to true or false for a description id
def set_acronym(request):
	adid = request.POST["adid"]
	is_acronym = True if request.POST["set-acronym"] == "true" else False
	alert = pg.set_acronym(adid, is_acronym)
	div = request.POST["results-div"]
	return JsonResponse(get_payload(div, alert=alert))


def set_concept_type(request):
	acid = request.POST["acid"]
	concept_type = request.POST["concept-type"]
	state = request.POST["state"]

	alert = ""

	# Catch casting errors for state
	if not state.isnumeric():
		alert = "Invalid value for state. Expected numeric string 0-3."
	else:
		state = int(state)

	if alert == "":
		alert += pg.set_concept_type(acid, concept_type, int(state))

	div = request.POST["results-div"]
	return JsonResponse(get_payload(div, alert=alert))



# Generates the html payload using the request and alert
# generated from pglib output
def get_payload(div, df=None, alert=None, header=None):
	data = df.to_dict("records") if df is not None else None
	html = render_to_string("graph/graph_table.html", {"payload" : data, "alert" : alert, "header" : header})
	payload = {}
	payload["data"] = [{"results-div" : div, "html" : html}]
	return payload

def post_concept_override(request):
	print("TEST")
	conn,cursor = pg.return_postgres_cursor()
	payload_dict = {}
	adid = ""
	acid = "" 
	
	if 'acid_relationship' in request.POST:
		acid = request.POST['acid_relationship']
		

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
		elif 'rel_chemical' in request.POST:
			rel_type = 'chemical'
		elif 'rel_outcome' in request.POST:
			rel_type = 'outcome'

		# 0 = inactive
		# 1 = active
		# 2 = too broad to display and too broad to use for training
		# 3 = too broad to display only
		if 'rel_activate' in request.POST:
			state = 1
		elif 'rel_inactivate' in request.POST:
			state = 0
		elif 'rel_broad_display_and_train' in request.POST:
			state = 2
		elif 'rel_broad_display_only' in request.POST:
			state = 3

		if rel_type is not None and state is not None:
			message = u.change_concept_type(acid, rel_type, state, cursor)
			payload_dict['change_concept_type_message'] = message
		else:
			payload_dict['change_concept_type_message'] = "Error status or rel_type is null"
		

	cursor.close()
	conn.close()

	return render(request, 'annotations/concept_override.html', {'payload' : payload_dict, 'acid' : acid, 'adid' : adid})
