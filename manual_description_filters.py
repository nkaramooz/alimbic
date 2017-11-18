import pandas as pd
import re
import psycopg2
from sqlalchemy import create_engine
from fuzzywuzzy import fuzz
import utilities.pglib as pg
import numpy as np
import time
import utilities.utils as u 


# engine = pg.return_sql_alchemy_engine()
	# results_df.to_sql('doc_annotation', engine, schema='annotation', if_exists='append')

DESC_ID_BLACKLIST = pd.DataFrame([ \
	['c86c4c65-a719-446a-9717-a62b1ac05cd4'] \
	,['2b052ceb-6bfe-4ea7-a96f-524fecd61529']\
	,['21857010'] \
	], columns=['id'])
DESCRIPTION_WHITELIST = pd.DataFrame([ \
	['42343007', 'CHF'] \
	,['10784006', 'Antipsychotics'] \
	,['409651001', 'Mortality'] \
	,['108661002', 'H2 antagonist'] \
	,['700372006', 'Catheter associated urinary tract infection']], columns=['conceptid', 'term'])


# NOTE: Still need to run psql file to update the
# augmented_concept_descriptions table based on 
# new filters
def update_filters_tables():
	engine = pg.return_sql_alchemy_engine()

	# NOTE - at some point write test such that blacklist and whitelist
	# don't collide
	DESC_ID_BLACKLIST.to_sql('description_id_blacklist', engine, schema='annotation', if_exists='replace')

	DESCRIPTION_WHITELIST.to_sql('description_whitelist', engine, schema='annotation', if_exists='replace')


def extract_value(string):
	res = []
	greater = False
	less = False
	trim = False
	prev_char = ''
	for char in string:
		if char == '>' and prev_char != ' ' and prev_char != '=':
			greater = True
		elif char == '<' and prev_char != ' ' and prev_char != '=':
			less = True
		elif char == ' ':
			greater = False
			less = False
			res.append(char)
		else:
			res.append(char)

		if greater and less:
			trim = True
		prev_char = char
	if trim:
		return ''.join(res)
	else:
		return string

def active_cleaned_selected_concept_descriptions_prelim():
	cursor = pg.return_postgres_cursor()
	desc_query = "select * from annotation.active_selected_concept_descriptions"
	desc_df = pg.return_df_from_query(cursor, desc_query, None, ["id", "conceptid", "term"])

	desc_df['term'] = desc_df['term'].map(extract_value)

	engine = pg.return_sql_alchemy_engine()
	desc_df.to_sql('active_cleaned_selected_concept_descriptions_prelim', engine, \
		schema='annotation', if_exists='replace')

def active_cleaned_distinct_terms():
	cursor = pg.return_postgres_cursor()
	engine = pg.return_sql_alchemy_engine()
	dist_query = "select distinct on (conceptid, term) \
		id, conceptid, term from annotation.active_cleaned_selected_concept_descriptions"
	dist_df = pg.return_df_from_query(cursor, dist_query, None, ["id", "conceptid", "term"])
	print("query complete")
	dist_df.to_sql('active_cleaned_selected_concept_descriptions', engine, \
		schema='annotation', if_exists='replace')


if __name__ == "__main__":
	active_cleaned_selected_concept_descriptions_prelim()