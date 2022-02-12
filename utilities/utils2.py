import time
import pandas as pd
import utilities.pglib as pg
import sqlalchemy as sqla
from nltk.stem.wordnet import WordNetLemmatizer
from elasticsearch import Elasticsearch, RequestsHttpConnection

def get_es_client():
	es = Elasticsearch(hosts=[{'host': 'vpc-elasticsearch-ilhv667743yj3goar2xvtbyriq.us-west-2.es.amazonaws.com', 'port' : 443}], use_ssl=True, verify_certs=True, connection_class=RequestsHttpConnection)
	# es = Elasticsearch([{'host' : 'localhost', 'port' : 9200, 'timeout' : 1000}])
	return es

class Timer:
	def __init__(self, label):
		self.label = label
		self.start_time = time.time()

	def stop(self):
		self.end_time = time.time()
		label = self.label + " : " + str(self.end_time - self.start_time)
		print(label)

	def stop_num(self):
		self.end_time = time.time()
		return float(self.end_time - self.start_time)

def pprint(data_frame):
	if isinstance(data_frame, str):
		print(data_frame)
	else:
		with pd.option_context('display.max_rows', None, 'display.max_columns', 10):
			pd.set_option('display.width', 1000)
			print(data_frame)

def get_conceptid_name(conceptid, cursor):
	search_query = """
		select 
			term
		from annotation2.preferred_concept_names
		where conceptid = '%s'
	""" % conceptid

	name = pg.return_df_from_query(cursor, search_query, None, ['term'])['term'].to_string(index=False)
	return name

def add_new_concept(concept_name, cursor):
	error = False
	conflict_df = check_existing_concept(concept_name, cursor)
	
	if len(conflict_df.index) > 0:
		error = True
		message = "Matches found in upstream_root_did. New concept not created."
	else:
		inactive_conflict_df = check_inactive_concepts(concept_name, cursor)
		if len(inactive_conflict_df.index) > 0:
			if len(inactive_conflict_df['acid'].unique()) == 1:
				message = "Activating previously inactived concept"
				existing_acid = inactive_conflict_df['acid']
				error = False
				query = """
					insert into annotation2.inactive_concepts (acid, active, effectivetime)
						VALUES(%s, 't', now())
					"""
				cursor.execute(query, (existing_acid,))
				cursor.connection.commit()
			else:
				message = "Multiple inactive concepts match this description"
				error = True
		else:
			query = """
				insert into annotation2.new_concepts (cid, did, term, effectivetime)
					VALUES(public.uuid_generate_v4(), public.uuid_generate_v4(), %s, now())
			"""
			cursor.execute(query, (concept_name,))
			cursor.connection.commit()
			message = "success"
			## figure out what to do about running lemmatizer

	return error, message, conflict_df

def remove_concept(acid, cursor):
	error = "Complete"

	try:
		query = """
			insert into annotation2.inactive_concepts (acid, active, effectivetime)
			VALUES (%s, 'f', now())
		"""	
		cursor.execute(query, (acid,))
		cursor.connection.commit()
	except:
		error = "Error"

	return error

# Make sure new description does not already exist
# If not, add to root_new_desc
def add_new_description(acid, new_description, cursor):
	error = False
	message = "success"

	# First check to make sure acid is correct
	query = """
		select 
			acid
		from annotation2.upstream_root_did
		where acid = %s
	"""
	acid_df = pg.return_df_from_query(cursor, query, (acid,), ['acid'])

	if len(acid_df.index) == 0:
		error = True
		message = "ACID not found in upstream_root_cid"
	else:
		# Now make sure new_description doesn't already exist

		query = """
			select 
				acid
				,adid
				,term
			from annotation2.lemmas
			where term = %s and acid=%s
		"""
		desc_df = pg.return_df_from_query(cursor, query, (new_description,acid), ['acid', 'adid', 'term'])

		if len(desc_df.index) > 0:
			message = "Description exists. ACID match : "
			acid_arr = desc_df['acid'].unique().tolist()
			message = ' '.join(acid_arr)
		else:	
			# Now you know the acid exists and description does not exist
			acid = int(acid)
			query = """
				insert into annotation2.root_new_desc (did,acid,term,active,effectivetime)
					VALUES(public.uuid_generate_v4(), %s, %s,'t', now())
			"""
			cursor.execute(query, (acid, new_description))
			cursor.connection.commit()
	return error, message

def change_relationship(child_acid, parent_acid, rel_action, cursor):
	error = False
	message = "success"
	if check_acid(child_acid, cursor) and check_acid(parent_acid, cursor):
		child_cid = get_cid_from_acid(child_acid, cursor)
		parent_cid = get_cid_from_acid(parent_acid, cursor)
		if child_cid is None or parent_cid is None:
			error = True
			message = "could not convert acid to CID in downstream table"
		else:
			query = """
				insert into snomed2.custom_relationship
				VALUES(public.uuid_generate_v4(), %s, %s, '116680003', %s, now())
			"""
			cursor.execute(query, (child_cid, parent_cid, rel_action))
			cursor.connection.commit()
	else:
		error = True
		message = "child or parent acid not found in root.downstream_cid"
	return error, message

def get_cid_from_acid(acid, cursor):
	query = "select cid from annotation2.downstream_root_cid where acid = %s"
	df = pg.return_df_from_query(cursor,query, (acid,), ['acid'])
	if len(df.index) > 0:
		return df['acid'][0]
	else:
		return None

def check_acid(acid, cursor):
	query = """
		select
			cid
		from annotation2.downstream_root_cid
		where acid = %s
	"""
	df = pg.return_df_from_query(cursor,query, (acid,), ['acid'])

	if len(df.index) > 0:
		return True
	else:
		return False

def change_concept_type(acid, rel_type, state, cursor):
	if check_acid(acid, cursor):
		query = """
			insert into annotation2.concept_types_app
				VALUES(%s, %s, %s, now())
		"""
		cursor.execute(query, (acid, rel_type, state))
		cursor.connection.commit()
		sql_file = open('/home/nkaramooz/Documents/alimbic/psql_files/annotation2_psql/concept_types/concept_types.sql', 'r')
		cursor.execute(sql_file.read())
		cursor.connection.commit()
		return "that probably worked"
	else:
		return "error, acid is not real"

	


def remove_adid(adid, cursor):
	# first check to make sure adid exists
	error = "No error"
	adid_df = get_existing_adid(adid, cursor)

	if len(adid_df.index) > 0:
		query = """
			insert into annotation2.root_desc_inactive(adid,term,active,effectivetime)
				select
					adid
					,term
					,'f' as active
					,now() as effectivetime
				from annotation2.lemmas
				where adid = %s
		"""
		cursor.execute(query, (adid,))
		cursor.connection.commit()
	else:
		error = "Error"
	return error

def add_labelled_treatment(condition_acid, treatment_acid, relationship, cursor):
	if condition_acid != None and treatment_acid != None and relationship != None:
		if (condition_acid == '%' and check_acid(treatment_acid)) or \
			check_acid(condition_acid, cursor) and check_acid(treatment_acid, cursor):

			if not check_existing_labelled_treatments(condition_acid, treatment_acid, cursor):
				query = """
					insert into ml2.labelled_treatments_app
					VALUES(public.uuid_generate_v4(), %s, %s, %s)
				"""
				cursor.execute(query, (condition_acid, treatment_acid, relationship))
				cursor.connection.commit()
				return "success"
			else:
				return "labelled_treatment exists"
		else:
			return "condition_acid or treatment_acid invalid"
	# Case when labelling condition to be too broad
	else:
		return "condition, treatment, or relationship not filled"

def check_existing_labelled_treatments(condition_acid, treatment_acid, cursor):
	query = """
		select condition_acid from ml2.labelled_treatments where condition_acid = %s and treatment_acid = %s
	"""
	db_df = pg.return_df_from_query(cursor, query, (condition_acid, treatment_acid), ['condition_acid'])

	if len(db_df.index) == 0:
		query = """
			select condition_acid from ml2.labelled_treatments_app where condition_acid = %s and treatment_acid = %s
		"""
		app_df = pg.return_df_from_query(cursor, query, (condition_acid, treatment_acid), ['condition_acid'])

		if len(app_df.index) == 0:
			return False
		else:
			return True
	else:
		return True

def get_existing_adid(adid, cursor):
	query = """
		select 
			adid
		from annotation2.lemmas
		where adid = %s
	"""
	adid_df = pg.return_df_from_query(cursor, query, (adid,), ['adid'])
	return adid_df

def check_existing_concept(concept_name, cursor):
	query = """
		select 
			acid
			,adid
			,term
		from annotation2.downstream_root_did
		where term = %s
	"""
	conflict_df = pg.return_df_from_query(cursor, query, (concept_name,), ['acid', 'adid', 'term'])

	return conflict_df

def check_inactive_concepts(concept_name, cursor):
	query = """
		select 
			t1.acid
			,t2.adid
			,t2.term
		from annotation2.inactive_concepts t1
		left join annotation2.upstream_root_did t2
		on t1.acid = t2.acid
		where t2.term = %s
	"""
	conflict_df = pg.return_df_from_query(cursor, query, (concept_name,), ['acid', 'adid', 'term'])

	return conflict_df


# Will need to rerun pipeline for update to propagate
def acronym_override(adid, is_acronym, cursor):
	
	df = get_existing_adid(adid, cursor)

	if len(df.index) > 0:
		modify_acronym_query = """
			set schema 'annotation';
			INSERT INTO acronym_override (id, adid, is_acronym, effectivetime)
			VALUES (public.uuid_generate_v4(), %s, %s, now())
		"""
		cursor.execute(modify_acronym_query, (adid, is_acronym))
		cursor.connection.commit()
		return "added to acronym_override"
	else:
		return "adid invalid"




def modify_concept_type(root_cid, associated_cid, new_rel_type, old_rel_type, cursor):
	remove_query = """
		set schema 'annotation';
		INSERT INTO concept_types (root_cid, associated_cid, rel_type, active, effectivetime)
		VALUES (%s, %s, %s, %s, now())
	"""
	cursor.execute(remove_query, (root_cid,associated_cid, old_rel_type, 0))
	cursor.connection.commit()

	remove_query = """
		set schema 'annotation';
		INSERT INTO override_concept_types (root_cid, associated_cid, rel_type, active, effectivetime)
		VALUES (%s, %s, %s, %s, now())
	"""
	cursor.execute(remove_query, (root_cid,associated_cid, old_rel_type, 0))
	cursor.connection.commit()

	insert_query = """
		set schema 'annotation';
		INSERT INTO concept_types (root_cid, associated_cid, rel_type, active, effectivetime)
		VALUES (%s, %s, %s, %s, now())
	"""
	cursor.execute(insert_query, (root_cid,associated_cid, new_rel_type, 1))
	cursor.connection.commit()

	insert_query = """
		set schema 'annotation';
		INSERT INTO override_concept_types (root_cid, associated_cid, rel_type, active, effectivetime)
		VALUES (%s, %s, %s, %s, now())
	"""

	cursor.execute(insert_query, (root_cid,associated_cid, new_rel_type, 1))
	cursor.connection.commit()

	return True

def treatment_label(condition_id, treatment_id, treatment_label, cursor):
	query = """
		set schema 'annotation';
		INSERT INTO labelled_treatments_app (condition_id, treatment_id, label)
		VALUES (%s, %s, %s)
	"""
	cursor.execute(query, (condition_id, treatment_id, treatment_label))
	cursor.connection.commit()

	query = """
		set schema 'annotation';
		CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
		INSERT INTO labelled_treatments (id, condition_id, treatment_id, label, ver)
		VALUES (public.uuid_generate_v4(), %s, %s, %s, 0)
	"""
	cursor.execute(query, (condition_id, treatment_id, treatment_label))
	cursor.connection.commit()
	print("COMMIT")



def add_concept(description, cursor):

	if not description_does_exist(description, cursor):
		insert_query = """
			set schema 'annotation';
			CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

			INSERT INTO new_concepts (conceptid, description_id, description, effectivetime)
			VALUES (public.uuid_generate_v4()::text, public.uuid_generate_v4()::text, %s, now())
			RETURNING conceptid, description_id;
		"""
		cursor.execute(insert_query, (description,))
		all_ids = cursor.fetchall()
		conceptid = all_ids[0][0]
		description_id = all_ids[0][1]
		tmp = "conceptid: " + conceptid + " description_id: " + description_id

		cursor.connection.commit()

		insert_query = """
			set schema 'annotation';
			INSERT into augmented_selected_concept_descriptions (description_id, conceptid, term, active, effectivetime)
			VALUES (%s, %s, %s,'1'::text, now())
		"""
		cursor.execute(insert_query, (description_id, conceptid, description))
		cursor.connection.commit()

		### augmented_active_selected_concept_key_words_v2
		insert_description_id_into_key_words_v3(conceptid, description_id, description, cursor)

		### augmented_active_selected_concept_key_words_lemmas_2
		lemmatize_description_id_3(description_id, cursor)
	else:
		raise ValueError("possible duplicate")





if __name__ == "__main__":
	print("no function")


