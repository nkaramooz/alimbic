import time
import pandas as pd
import utilities.pglib as pg
from nltk.stem.wordnet import WordNetLemmatizer
from elasticsearch import Elasticsearch, RequestsHttpConnection

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
	with pd.option_context('display.max_rows', None, 'display.max_columns', 10):
		pd.set_option('display.width', 1000)
		print(data_frame)

def write_sentences(s_df, cursor):
	engine = pg.return_sql_alchemy_engine()
	s_df.to_sql('sentences2', \
		engine, schema='annotation', if_exists='append', index=False)

def get_conceptid_name(conceptid, cursor):
	search_query = """
		select 
			term
		from annotation.preferred_concept_names
		where conceptid = '%s'
	""" % conceptid

	name = pg.return_df_from_query(cursor, search_query, None, ['term'])['term'].to_string(index=False)
	return name

def log_annotation_error(root_cid, associated_cid, old_rel_type, cursor):
	log_query = """
		set schema 'annotation';
		INSERT INTO annotation_errors (root_cid, associated_cid, effectivetime)
		VALUES(%s, %s, now())
	"""

	cursor.execute(log_query, (root_cid, associated_cid))
	cursor.connection.commit()

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

	return True

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

def add_description(conceptid, description, cursor):

	if is_valid_conceptid(conceptid, cursor):
		if not is_duplicate_description_for_concept(conceptid, description, cursor):

			### annotation.augmented_selected_concept_descriptions
			description_id = insert_new_description_into_aug_sel_concept_desc(conceptid, \
				description, cursor)


			### augmented_active_selected_concept_key_words_v2

			insert_description_id_into_key_words_v3(conceptid, description_id, description, \
				cursor)

			### augmented_active_selected_concept_key_words_lemmas_2
			lemmatize_description_id_3(description_id, cursor)

			### update_white_list
			insert_into_whitelist(conceptid, description_id, description, cursor)
		else:
			raise ValueError("description exists")
	else:
		raise ValueError('Invalid conceptid')

def activate_description_id(description_id, cursor):
	### annotation.augmented_selected_concept_descriptions
	insert_query = """
			set schema 'annotation';
			INSERT INTO augmented_selected_concept_descriptions 
				(description_id, conceptid, term, active, effectivetime)
			select 
				description_id
				,conceptid
				,term
				,'1'::text as active
				,now() as effectivetime
			from annotation.augmented_selected_concept_descriptions
			where description_id = %s
			limit 1;
	"""
	try:
		cursor.execute(insert_query, (description_id,))
		cursor.connection.commit()
	except:
		raise ValueError("unable to description_id into augmented_selected_concept_descriptions")
	
	conceptid, description = get_conceptid_and_description_from_id(description_id, cursor)
	
	### key words table
	try:
		insert_description_id_into_key_words_v3(conceptid, description_id, description, cursor)
	except:
		raise ValueError("unable to add description_id to key words table")

	### augmented_active_selected_concept_key_words_lemmas_2
	lemmatize_description_id_3(description_id, cursor)

	if is_duplicate_description_id_for_table(description_id, \
		"description_blacklist", cursor):
		print("ON BLACKLIST")
		delete_description_id_from_table(description_id, "description_blacklist", cursor)

def get_conceptid_and_description_from_id(description_id, cursor):
	query = "select conceptid, term from annotation.augmented_selected_concept_descriptions \
		where description_id = %s limit 1"
	df = pg.return_df_from_query(cursor, query, (description_id,), ['conceptid', 'term'])

	conceptid = df.iloc[0]['conceptid']
	description = df.iloc[0]['term']
	return conceptid,description

def deactivate_description_id(description_id, cursor):

	if is_duplicate_description_id_for_table(description_id, \
		"augmented_selected_concept_descriptions", cursor):

		delete_query = """
			set schema 'annotation';
			INSERT INTO augmented_selected_concept_descriptions 
				(description_id, conceptid, term, active, effectivetime)
			select 
				description_id
				,conceptid
				,term
				,'0'::text as active
				,now() as effectivetime
			from annotation.augmented_selected_concept_descriptions
			where description_id = %s
			limit 1;
		"""

		cursor.execute(delete_query, (description_id,))
		cursor.connection.commit()


		delete_description_id_from_table(description_id, \
			"augmented_active_key_words_v3", cursor)
		
		delete_description_id_from_table(description_id, \
			"lemmas_3", cursor)


		if is_duplicate_description_id_for_table(description_id, \
			"description_whitelist", cursor):
			delete_description_id_from_table(description_id, "description_whitelist", cursor)
		else:
			conceptid, description = get_conceptid_and_description_from_id(description_id, cursor)
			insert_into_blacklist(conceptid, description_id, description, cursor)

	else:
		raise ValueError('Invalid description_id')

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

#### HELPER FUNCTIONS
def description_does_exist(description, cursor):
	query = "select conceptid from annotation.augmented_selected_concept_descriptions\
		where term = %s"

	df = pg.return_df_from_query(cursor, query, (description,), ['conceptid'])
	if len(df) > 0:
		return True
	else:
		return False

def is_valid_conceptid(conceptid, cursor):

	query = "select conceptid from annotation.augmented_selected_concept_descriptions where \
		conceptid = '%s' " % conceptid 

	if len(pg.return_df_from_query(cursor, query, None, ["conceptid"])) > 0:
		return True
	else:
		return False

def is_duplicate_description_for_concept(conceptid, description, cursor):
	query = "select conceptid from annotation.augmented_selected_concept_descriptions \
		where conceptid = %s and term = %s"

	if len(pg.return_df_from_query(cursor, query, (conceptid, description), ["conceptid"])) > 0:
		return True
	else:
		return False

def is_duplicate_description_id_for_table(description_id, table_name, cursor):

	check_query = "set schema 'annotation'; select description_id from " 
	check_query += table_name + " where description_id = %s"
	if len(pg.return_df_from_query(cursor, check_query, (description_id,), ['description_id'])) > 0:
		return True
	else:
		return False
	return False

def insert_new_description_into_aug_sel_concept_desc(conceptid, description, cursor):
	insert_query = """
			set schema 'annotation';
			CREATE EXTENSION IF NOT EXISTS "uuid-ossp" ;

			INSERT INTO augmented_selected_concept_descriptions (description_id, conceptid, term, active, effectivetime)
			VALUES (public.uuid_generate_v4()::text, %s, %s, '1'::text, now()) RETURNING description_id;
	"""

	cursor.execute(insert_query, (conceptid, description))
	description_id = cursor.fetchone()[0]
	cursor.connection.commit()
	return description_id

def insert_description_id_into_key_words_v2(conceptid, description_id, description, cursor):
	insert_query = get_key_word_query(conceptid, description_id, description)
	cursor.execute(insert_query, None)
	cursor.connection.commit()

def insert_description_id_into_key_words_v3(conceptid, description_id, description, cursor):
	insert_query = get_key_word_query_3(conceptid, description_id, description)
	cursor.execute(insert_query, None)
	cursor.connection.commit()

def insert_into_whitelist(conceptid, description_id, description, cursor):
	insert_query = """
			set schema 'annotation';
			CREATE TABLE IF NOT EXISTS description_whitelist (description_id text, conceptid text, term text);
			INSERT INTO description_whitelist (description_id, conceptid, term)
			VALUES (%s, %s, %s);
		"""
	cursor.execute(insert_query, (description_id, conceptid, description))
	cursor.connection.commit()

def insert_into_blacklist(conceptid, description_id, description, cursor):
	insert_query = """
			set schema 'annotation';
			CREATE TABLE IF NOT EXISTS description_blacklist (description_id text, conceptid text, term text);
			INSERT INTO description_blacklist (description_id, conceptid, term)
			VALUES (%s, %s, %s);
		"""
	cursor.execute(insert_query, (description_id, conceptid, description))
	cursor.connection.commit()

def delete_description_id_from_table(description_id, table_name, cursor):
	delete_query = "set schema 'annotation'; DELETE FROM "
	delete_query += table_name + " where description_id = %s"
	cursor.execute(delete_query, (description_id,))
	cursor.connection.commit()

def get_key_word_query(conceptid, descriptionid, description):
	query = """
		set schema 'annotation';
		INSERT INTO augmented_active_selected_concept_key_words_v2 (description_id, conceptid, term, word, word_ord, term_length)

		select 
  			concept_table.description_id
  			,concept_table.conceptid
  			,concept_table.term
  			,concept_table.word
  			,concept_table.word_ord
  			,len_tb.term_length
		from (
    		select 
            	description_id
            	,conceptid
            	,term
            	,case when upper(word) = word then word else lower(word) end as word
            	,word_ord
    		from (
        		select 
            		description_id
            		,conceptid
            		,term
            		,word
            		,word_ord
        		from (
            		select 
                    	'%s'::text as description_id
                    	,'%s'::text as conceptid
                    	,'%s'::text as term
               	) tb, unnest(string_to_array(replace(replace(replace(tb.term, ' - ', ' '), '-', ' '), ',', ''), ' '))
                    with ordinality as f(word, word_ord)
        	) nm
      		where lower(word) not in (select lower(words) from annotation.filter_words)
  		) concept_table
		join (
    		select
     			description_id
      			,count(*) as term_length
    		from (
    		select 
          		description_id
          		,conceptid
          		,term
          		,word
        	from (
    			select
    				description_id
    				,conceptid
    				,term
    				,lower(unnest(string_to_array(replace(replace(replace(term, ' - ', ' '), '-', ' '), ',', ''), ' '))) as word
    			from (
					select 
                    	'%s'::text as description_id
                    	,'%s'::text as conceptid
                    	,'%s'::text as term
               		) tb 
            	) tb2
    			where lower(word) not in (select lower(words) from annotation.filter_words)
       		) tmp
    		group by tmp.description_id
   		) len_tb
 		on concept_table.description_id = len_tb.description_id
 	; 
	""" % (descriptionid, conceptid, description, descriptionid, conceptid, description)

	return query

def get_key_word_query_3(conceptid, descriptionid, description):
	query = """
		set schema 'annotation';
		INSERT INTO augmented_active_key_words_v3 (description_id, conceptid, term, word, word_ord, term_length)

		select 
  			concept_table.description_id
  			,concept_table.conceptid
  			,concept_table.term
  			,concept_table.word
  			,concept_table.word_ord
  			,len_tb.term_length
		from (
    		select 
            	description_id
            	,conceptid
            	,term
            	,case when upper(word) = word then word else lower(word) end as word
            	,word_ord
    		from (
        		select 
            		description_id
            		,conceptid
            		,term
            		,word
            		,word_ord
        		from (
            		select 
                    	'%s'::text as description_id
                    	,'%s'::text as conceptid
                    	,'%s'::text as term
               	) tb, unnest(string_to_array(replace(replace(replace(tb.term, ' - ', ' '), '-', ' '), ',', ''), ' '))
                    with ordinality as f(word, word_ord)
        	) nm
  		) concept_table
		join (
    		select
     			description_id
      			,count(*) as term_length
    		from (
    		select 
          		description_id
          		,conceptid
          		,term
          		,word
        	from (
    			select
    				description_id
    				,conceptid
    				,term
    				,lower(unnest(string_to_array(replace(replace(replace(term, ' - ', ' '), '-', ' '), ',', ''), ' '))) as word
    			from (
					select 
                    	'%s'::text as description_id
                    	,'%s'::text as conceptid
                    	,'%s'::text as term
               		) tb 
            	) tb2
       		) tmp
    		group by tmp.description_id
   		) len_tb
 		on concept_table.description_id = len_tb.description_id
 	; 
	""" % (descriptionid, conceptid, description, descriptionid, conceptid, description)

	return query

def lemma(word):
	lmtzr = WordNetLemmatizer()
	return lmtzr.lemmatize(word)

def lemmatize_description_id_3(description_id, cursor):
	query = """
		select * from annotation.augmented_active_key_words_v3
		where description_id = '%s'
	""" % description_id

	new_candidate_df = pg.return_df_from_query(cursor, query, None, \
		['description_id', 'conceptid', 'term', 'word', 'word_ord', 'term_length'])
	new_candidate_df['word'] = new_candidate_df['word'].map(lemma)

	engine = pg.return_sql_alchemy_engine()
	new_candidate_df.to_sql('lemmas_3', \
		engine, schema='annotation', if_exists='append', index=False)

def lemmatize_table():
	query = "select * from annotation.augmented_active_key_words_v3"
	conn, cursor = pg.return_postgres_cursor()

	new_candidate_df = pg.return_df_from_query(cursor, query, None, \
		['description_id', 'conceptid', 'term', 'word', 'word_ord', 'term_length'])

	# don't want As to lemmatize to "a"
	new_candidate_df.loc[~new_candidate_df.word.isin(['vs', 'as']), 'word'] = new_candidate_df.loc[~new_candidate_df.word.isin(['vs', 'as'])]['word'].map(lemma)
	engine = pg.return_sql_alchemy_engine()

	new_candidate_df.to_sql('lemmas_3', \
		engine, schema='annotation', if_exists='replace', index=False)

	index_query = """
		set schema 'annotation';
		create index lemmas3_conceptid_ind on lemmas_3(conceptid);
		create index lemmas3_description_id_ind on lemmas_3(description_id);
		create index lemmas3_term_ind on lemmas_3(term);
		create index lemmas3_word_ind on lemmas_3(word);
		create index lemmas3_word_ord_ind on lemmas_3(word_ord);
	"""

	cursor.execute(index_query, None)
	cursor.connection.commit()
	cursor.close()

def insert_new_vc_user(name, cursor):
	insert_query = """
			set schema 'vancocalc';
			CREATE EXTENSION IF NOT EXISTS "uuid-ossp" ;

			INSERT INTO users (uid, username, active, effectivetime)
			VALUES (public.uuid_generate_v4(), %s, 1, now());
	"""

	cursor.execute(insert_query, (name,))
	cursor.connection.commit()
	return True

def insert_new_vc_case(uid, casename, cursor):
	insert_query = """
			set schema 'vancocalc';
			CREATE EXTENSION IF NOT EXISTS "uuid-ossp" ;

			INSERT INTO cases (cid, uid, casename, active, effectivetime)
			VALUES (public.uuid_generate_v4(), %s, %s, 1, now())
			RETURNING cid;
	"""
	cursor.execute(insert_query, (uid,casename))
	ids = cursor.fetchall()
	cid = ids[0][0]
	cursor.connection.commit()
	return cid

def get_es_client():
	# es = Elasticsearch(hosts=[{'host': 'vpc-elasticsearch-ilhv667743yj3goar2xvtbyriq.us-west-2.es.amazonaws.com', 'port' : 443}], use_ssl=True, verify_certs=True, connection_class=RequestsHttpConnection)
	es = Elasticsearch([{'host' : 'localhost', 'port' : 9200}])
	return es

if __name__ == "__main__":
	lemmatize_table()


