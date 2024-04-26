import psycopg2.pool
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import URL
import os

dbpool = psycopg2.pool.ThreadedConnectionPool(2,10, host=os.environ.get("ALIMBIC_DB_HOST"), \
	dbname=os.environ["ALIMBIC_DB"], user=os.environ["USER"])

def get_conn():
	conn = dbpool.getconn()
	return conn


def return_postgres_cursor():
	conn = dbpool.getconn()
	cursor = conn.cursor()
	return conn,cursor


def return_pg_conn():
	conn = dbpool.getconn()
	return conn


# Returns dataframe from query
# Empty list returned if no results found
def return_df_from_query(sql_query, params, column_names):
	conn = dbpool.getconn()
	cursor = conn.cursor()
	cursor.execute(sql_query, params)
	records = cursor.fetchall()
	dbpool.putconn(conn)
	return pd.DataFrame(records, columns = column_names)


def return_sql_alchemy_engine():
	url_object = URL.create(
		"postgresql",
		username=os.environ["USER"],
		password=os.environ.get("ALIMBIC_DB_PSWD"),
		host=os.environ.get("ALIMBIC_DB_HOST"),
		database=os.environ.get("ALIMBIC_DB"))
	engine = create_engine(url_object)
	return engine


# Returns true if the write was successful
def write_data(query, params):
	try:
		conn = dbpool.getconn()
		cursor = conn.cursor()
		cursor.execute(query, params)
		# cursor.connection.commit()
		dbpool.putconn(conn)
		return True
	except:
		return False


def return_numpy_from_query(cursor, sql_query, params):
	cursor.execute(sql_query, params)
	records = np.array(cursor.fetchall())
	return records


# Input must be string.
# Returns true if the acid exists in the graph.
def acid_exists(acid):
	query = """
		SELECT
			acid
		FROM annotation2.downstream_root_cid
		WHERE acid = %s
	"""
	df = return_df_from_query(query, (acid,), ["acid"])
	return True if len(df.index) > 0 else False


# Input must be string.
# Returns true if the adid exists in the graph.
def adid_exists(adid):
	query = """
		SELECT
			adid
		FROM annotation2.add_adid_acronym
		WHERE adid = %s
	"""
	df = return_df_from_query(query, (adid,), ["adid"])
	return True if len(df.index) > 0 else False


# Return True if description exists for acid.
def acid_description_exists(acid, term):
	query = """
			SELECT 
				acid
			FROM annotation2.add_adid_acronym
			WHERE acid=%s and term = %s
		"""
	df = return_df_from_query(query, (acid,term), ["acid"])
	return True if len(df.index) > 0 else False


# Returns true if the case sensitive term is a concept in the graph
# Note it does not need to be an active concept
def check_active_concepts(term):
	query = """
		SELECT 
			acid
			,adid
			,term
		FROM annotation2.downstream_root_did
		WHERE term = %s
	"""
	df = return_df_from_query(query, (term,), ['acid', 'adid', 'term'])
	return True if len(df.index) > 0 else False


# Evaluates the upstream table for potential matches on term
def check_inactive_concepts(term):
	query = """
		SELECT 
			t1.acid
			,t2.adid
			,t2.term
		FROM annotation2.inactive_concepts t1
		LEFT JOIN annotation2.upstream_root_did t2
			ON t1.acid = t2.acid
		WHERE t2.term = %s
	"""
	df = return_df_from_query(query, (term,), ['acid', 'adid', 'term'])
	return True if len(df.index) > 0 else False


# Assign a new relationship between a condition acid and treatment acid
# Condition acid is allowed to be the wildcard "%"
# Function returns an error for invalid IDs and existing relationships
def add_labelled_treatment(condition_acid, treatment_acid, rel):
	if rel not in ("0","1","2"):
		return "Relationship integer provided is not accepted."

	if condition_acid == "" or treatment_acid == "":
		return "Condition or treatment ID is empty" 
	elif not((acid_exists(condition_acid) or condition_acid == "%") and 
		acid_exists(treatment_acid)):
		return "ID for condition or treatment invalid."
	else:
		if check_existing_labelled_treatments(condition_acid, treatment_acid):
			return "Labelled relationship already exists."
		else:
			alert = write_labelled_treatment(condition_acid, treatment_acid, rel)
			return "Write successful." if alert else "Write unsuccessful. Check write_labelled_treatment function."


# Evaluates for an existing relationship between a condition and treatment
# It does not check the relationship type and does not check if a rule
# Applying to a parent concept exists. TODO: This creates the potential for 
# Rules conflicting due to the parent/child relationship of the graph
def check_existing_labelled_treatments(condition_acid, treatment_acid):
	query = """
		SELECT 
			condition_acid 
		FROM ml2.labelled_treatments 
		WHERE condition_acid = %s AND treatment_acid = %s
	"""
	df = return_df_from_query(query, (condition_acid, treatment_acid), ['condition_acid'])
	return True if len(df.index) != 0 else False


# Writes a new treatment relationship
def write_labelled_treatment(condition_acid, treatment_acid, rel):
	query = """
		INSERT INTO ml2.labelled_treatments_app
		VALUES(public.uuid_generate_v4(), %s, %s, %s)
	"""
	return write_data(query, (condition_acid, treatment_acid, rel))


# Input acid conceptid
# Return None if does not exist
def get_parents(acid):
	if not acid_exists(acid):
		return None
	else:
		query = """
				SELECT
					source_acid as item
					,t3.term as item_name
					,destination_acid as parent
					,t2.term as parent_name
				FROM snomed2.full_relationship_acid t1
				JOIN (select distinct on (acid) acid, term FROM annotation2.downstream_root_did) t2
					ON t1.destination_acid = t2.acid
				JOIN (select distinct on (acid) acid, term FROM annotation2.downstream_root_did) t3
					ON t1.source_acid = t3.acid
				WHERE source_acid = %s and typeid='116680003' and active='1'
			"""
		df = return_df_from_query(query, (acid,), ['item', 'item_name', 'parent', 'parent_name'])
		return df


# Input acid conceptid
# Return None if does not exist
def get_children(acid):
	if not acid_exists(acid):
		return None
	else:
		query = """
				SELECT
					source_acid as child
					,t2.term as child_name
					,destination_acid as item
					,t3.term as item_name
				FROM snomed2.full_relationship_acid t1
				JOIN (SELECT DISTINCT ON (acid) acid, term FROM annotation2.downstream_root_did) t2
					ON t1.source_acid = t2.acid
				JOIN (SELECT DISTINCT ON (acid) acid, term FROM annotation2.downstream_root_did) t3
					ON t1.destination_acid = t3.acid
				WHERE destination_acid = %s and typeid='116680003' and active='1'
			"""
		df = return_df_from_query(query, (acid,), ['child', 'child_name', 'item', 'item_name'])
		return df


# Retrieves matching concepts to the lowered term
def get_term(term):
	query = """
		SELECT 
			t1.adid
			,t1.acid
			,t2.cid
			,t1.term
			,t1.term_lower
			,t1.word
			,t1.word_ord
			,t1.is_acronym
		FROM annotation2.add_adid_acronym t1
		LEFT JOIN annotation2.downstream_root_cid t2
			ON t1.acid = t2.acid
		WHERE t1.term_lower = %s
	"""
	df = return_df_from_query(query, (term.lower(),), \
	 	["adid", "acid", "cid", "term", "term_lower",'word', "word_ord", "is_acronym"])
	return df


# Creates a new concept in the graph
# if the provider term does not match an existing term
def create_concept(term):
	if check_active_concepts(term):
		return "Matches found in downstream_root_did. New concept not created."
	elif check_inactive_concepts(term):
		return "Match found in upstream_root_did. New concept not created."
	else:
		query = """
				INSERT INTO annotation2.new_concepts (cid, did, term, effectivetime)
				VALUES(public.uuid_generate_v4(), public.uuid_generate_v4(), %s, now())
			"""
		return "Success writing new concept" if write_data(query, (term,)) else "Error writing concept"


# Marks a concept to be made inactive the next time the 
# knowledge graph is updated
def deactivate_concept(acid):
	if not acid_exists(acid):
		return "Unable to find requested conceptid"
	else:
		query = """
			INSERT INTO annotation2.inactive_concepts (acid, active, effectivetime)
			VALUES (%s, 'f', now())
		"""
		return "Success inactivating concept" if write_data(query, (acid,)) \
			else "Error inactivating concept. Check pglib function."



# Creates a new description attached to the concept id.
# Returns an alert message if the concept id does not exist
# or if the description already exists.
def add_description(acid, term):
	if not acid_exists(acid):
		return "Invalid concept id provided."
	elif acid_description_exists(acid, term):
		return "Description already exists for this concept id."
	else:
		acid = int(acid) # Required data type for root_new_desc

		query = """
				INSERT INTO annotation2.root_new_desc (did, acid, term, active, effectivetime)
					VALUES(public.uuid_generate_v4(), %s, %s,'t', now())
		"""
		return "Success adding description" if write_data(query, (acid, term)) else "Error writing new description. Check pglib function."


# Marks a description ID (ADID) to be made inactive
# the next time the knowledge graph is updated.
# Returns an alert message if the description id does not exist.
def deactivate_description(adid):
	if not adid_exists(adid):
		return "Invalid description id provided"
	else:
		query = """
			INSERT INTO annotation2.root_desc_inactive(adid,term,active,effectivetime)
				SELECT
					adid
					,term
					,'f' as active
					,now() as effectivetime
				FROM annotation2.add_adid_acronym
				WHERE adid = %s
		"""
		return "Success removing description." if write_data(query, (adid,)) \
			else "Error removing description. Check pglib function."


# Returns the snomed concept id (cid)
# from a provided acid if one exists, otherwise returns None
def get_cid_from_acid(acid):
	query = "SELECT cid FROM annotation2.downstream_root_cid WHERE acid = %s"
	df = return_df_from_query(query, (acid,), ['acid'])
	return df['acid'][0] if len(df.index) > 0 else None


# Marks a relationship between two acids as parent/child to be added or
# removed in the next rebuild of the knowledge graph.
# TODO: Alert user if request to delete a relationship that doesn't exist.
def modify_relationship(child_acid, parent_acid, add):
	if not (acid_exists(child_acid) and acid_exists(parent_acid)):
		return "Invalid child or parent concept id."
	elif add not in ("0", "1"):
		return "Invalid relationship value provided. Should be string '0' or '1'."
	else:
		child_cid = get_cid_from_acid(child_acid)
		parent_cid = get_cid_from_acid(parent_acid)

		if None in (child_cid, parent_cid):
			return "Unable to convert concept id provided to a snomed concept id."
		else:
			# 11668003 relationship in snomed refers to "is a"
			query = """
					INSERT INTO snomed2.custom_relationship
					VALUES(public.uuid_generate_v4(), %s, %s, '116680003', %s, now())
			"""
			return "Success modifying relationship." if write_data(query, (child_cid, parent_cid, add)) else \
				"Error changing relationship. Check pglib function."


# Setting a description's acronym flag to True
# means that in future annotations, that description will only
# be annotated if there is supporting data for that concept id.
# For example, CHF would only be annotated by heart failure
# if somewhere in the document "Congestive heart failure" appeared.
# is_acronym provided should be a boolean.
# Note that update will not take effect until knowledge graph is rebuilt.
def set_acronym(adid, is_acronym):
	if not adid_exists(adid):
		return "Invalid description id provided."
	elif type(is_acronym) != bool:
		return "Boolean not provided for is_acronym field."
	else:
		query = """
			INSERT INTO annotation2.acronym_override (id, adid, is_acronym, effectivetime)
			VALUES (public.uuid_generate_v4(), %s, %s, now())
		"""
		return "Success setting acronym status." if write_data(query, (adid, is_acronym)) else \
			"Error setting acronym status. Check pglib function."


# concept_type should be the exact strings listed.
# state is an integer with values 0-3 with definitions listed below.
def set_concept_type(acid, concept_type, state):
	if concept_type not in ("condition", "symptom", "treatment", 
		"cause", "diagnostic", "statistic", "chemical", "outcome"):
		return "Invalid concept type provided."

	if state not in (0, 1, 2, 3):
		return ("Invalid state value provided. Value should be an integer 0= deactivate," 
			"1 = activate, 2 = too broad to display and use for training, "
			"3 = too broad to display (will still be used for training)"
		)
		
	if not acid_exists(acid):
		return "Invalid concept id provided."

	query = """
			insert into annotation2.concept_types_app
			VALUES(%s, %s, %s, now())
		"""
	return "Success setting concept type" if write_data(query, (acid, concept_type, state)) else \
		"Error writing concept type. Check pglib function."


# Return all active condition/symptom concepts.
def get_all_conditions_set():
	query = """select root_acid from annotation2.concept_types 
		where (rel_type='condition' or rel_type='symptom') 
		and (active=1 or active=3)
	"""
	all_conditions_set = set(return_df_from_query(query, None, ['root_acid'])['root_acid'].tolist())
	return all_conditions_set


# Returns all active outcomes concepts.
def get_all_outcomes_set():
	query = """select root_acid from annotation2.concept_types 
		where (rel_type='outcome') 
		and (active=1 or active=3)
	"""
	all_outcomes_set = set(return_df_from_query(query, None, ['root_acid'])['root_acid'].tolist())
	return all_outcomes_set


# Returns all active statistic concepts.
def get_all_statistics_set():
	query = """select root_acid from annotation2.concept_types 
		where (rel_type='statistic') 
		and (active=1 or active=3)
	"""
	all_statistics_set = set(return_df_from_query(query, None, ['root_acid'])['root_acid'].tolist())
	return all_statistics_set


# Returns all active study design concepts.
def get_all_study_designs_set():
	query = """select root_acid from annotation2.concept_types 
		where (rel_type='study_design') 
		and (active=1 or active=3)
	"""
	all_study_designs_set = set(return_df_from_query(query, None, ['root_acid'])['root_acid'].tolist())
	return all_study_designs_set


# Return all active chemical concepts.
def get_all_chemicals_set():
	query = """select root_acid from annotation2.concept_types 
		where (rel_type='chemical') 
		and (active=1 or active=3)
	"""
	all_chemicals_set = set(return_df_from_query(query, None, ['root_acid'])['root_acid'].tolist())
	return all_chemicals_set


# Return all active treatment concepts.
def get_all_treatments_set():
	query = """
		select root_acid from annotation2.concept_types 
		where rel_type='treatment' 
		and (active=1 or active=3)
	"""
	all_treatments_set = set(return_df_from_query(query, None, ['root_cid'])['root_cid'].tolist())
	return all_treatments_set


# Return all active anatomy concepts.
def get_all_anatomy_set():
	query = """
		select root_acid from annotation2.concept_types 
		where rel_type='anatomy' 
		and (active=1 or active=3)
	"""
	all_anatomy_set = set(return_df_from_query(query, None, ['root_cid'])['root_cid'].tolist())
	return all_anatomy_set


# Return all treatment candidates, including ones marked as inactive.
def get_all_treatments_with_inactive():
	query = """
		select root_acid from annotation2.concept_types 
		where rel_type='treatment'
	"""
	all_treatments_set = set(return_df_from_query(query, None, ['root_cid'])['root_cid'].tolist())
	return all_treatments_set


# Returns all active causes concepts.
def get_all_causes_set():
	query = """
		select root_acid from annotation2.concept_types 
		where rel_type='cause' 
		and (active=1 or active=3)
	"""
	all_cause_set = set(return_df_from_query(query, None, ['root_cid'])['root_cid'].tolist())
	return all_cause_set


# Return all active diagnostic concepts.
def get_all_diagnostics_set():
	query = """
		select root_acid from annotation2.concept_types 
		where rel_type='diagnostic' 
		and (active=1 or active=3) 
	"""
	all_diagnostic_set = set(return_df_from_query(query, None, ['root_cid'])['root_cid'].tolist())
	return all_diagnostic_set


# Return all active concepts of interest.
def get_all_concepts_of_interest():
	concepts_of_interest = get_all_conditions_set()
	concepts_of_interest.update(get_all_treatments_set())
	concepts_of_interest.update(get_all_diagnostics_set())
	concepts_of_interest.update(get_all_causes_set())
	concepts_of_interest.update(get_all_outcomes_set())
	concepts_of_interest.update(get_all_statistics_set())
	concepts_of_interest.update(get_all_chemicals_set())
	concepts_of_interest.update(get_all_study_designs_set())
	return concepts_of_interest