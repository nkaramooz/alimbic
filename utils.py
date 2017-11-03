import time
import pandas as pd
import pglib as pg
import lemmatizer as lmtzr

class Timer:
	def __init__(self, label):
		self.label = label
		self.start_time = time.time()

	def stop(self):
		self.end_time = time.time()
		label = self.label + " : " + str(self.end_time - self.start_time)
		print(label)

def pprint(data_frame):
	with pd.option_context('display.max_rows', None, 'display.max_columns', 10):
		pd.set_option('display.width', 1000)
		print(data_frame)

def is_valid_conceptid(conceptid, cursor):

	query = "select conceptid from annotation.active_selected_concepts where \
		conceptid = '%s' " % conceptid 

	if len(pg.return_df_from_query(cursor, query, None, ["conceptid"])) > 0:
		return True
	else:
		return False

def is_duplicate_description_for_concept(conceptid, description, cursor):
	query = "select conceptid from annotation.augmented_active_selected_concept_descriptions \
		where conceptid = '%s' and term = '%s' " % (conceptid, description)

	if len(pg.return_df_from_query(cursor, query, None, ["conceptid"])) > 0:
		return True
	else:
		return False
 
def whitelist_description_for_conceptid(conceptid, description, cursor):
	if not is_duplicate_description_for_concept(conceptid, description, cursor):

		### annotation.augmented_selected_concept_descriptions
		insert_query = """
			set schema 'annotation';
			CREATE EXTENSION IF NOT EXISTS "uuid-ossp" ;

			INSERT INTO augmented_active_selected_concept_descriptions (id, conceptid, term, active, effectivetime)
			VALUES (public.uuid_generate_v4()::text, %s, %s, '1'::text, now()) RETURNING id;
		"""

		cursor.execute(insert_query, (conceptid, description))
		description_id = cursor.fetchone()[0]
		cursor.connection.commit()


		### augmented_active_selected_concept_key_words_v2
		insert_query = get_key_word_query(conceptid, description_id, description)
		cursor.execute(insert_query)
		cursor.connection.commit()


		### augmented_active_selected_concept_key_words_lemmas_2
		lmtzr.lemmatize_description_id(description_id, cursor)

		### update_white_list
		insert_query = """
			set schema 'annotation';
			CREATE TABLE IF NOT EXISTS description_whitelist (description_id, conceptid, term);
			INSERT INTO description_whitelist (description_id, conceptid, term)
			VALUES (%s, %s, %s);
		"""
		cursor.execute(insert_query, (description_id, conceptid, description))
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
            	,lower(word) as word
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

def is_existing_description_id(description_id, cursor):
	check_query = """
		select description_id from annotation.active_selected_concept_descriptions
		where description_id = '%s'
	"""

	if len(pg.return_df_from_query(cursor, check_query, (description_id,), ['description_id'])) > 0:
		return True
	else:
		return False

def delete_description_id(description_id, cursor):

	if is_existing_description_id(description_id, cursor):
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
			where id = '%s'
			limit 1;
		"""

		cursor.execute(delete_query, (description_id,))
		cursor.connection.commit()

		delete_query = """
			set schema 'annotation';
			DELETE FROM augmented_active_selected_concept_key_words_v2
			where description_id = '%s'
		"""

		cursor.execute(delete_query, (description_id,))
		cursor.connection.commit()

		delete_query = """
			set schema 'annotation';
			DELETE FROM augmented_active_selected_concept_key_words_lemmas_2
			where description_id = '%s'
		""" 

		cursor.execute(delete_query, (description_id,))
		cursor.connection.commit()
		
	else:
		raise ValueError('Invalid description_id')