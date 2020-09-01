import pandas as pd
import re
import psycopg2
from sqlalchemy import create_engine
from fuzzywuzzy import fuzz
import utilities.pglib as pg
import numpy as np
import time
import utilities.utils as u 


def extract_value(term):
	res = []
	greater = False
	less = False
	trim = False
	prev_char = ''
	for char in term:
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
		return term



# Immunoglobulin, GM>5< allotype (substance)
# Immunoglobulin, GM5 allotype (substance)
# end
# Third heart sound, S>3<, inaudible (finding)
# Third heart sound, S3, inaudible (finding)
# end
# First heart sound, S>1<
# First heart sound, S1
# end
# Hemoglobin A>2< Adria
# Hemoglobin A2 Adria
def active_cleaned_selected_concept_descriptions_prelim():
	conn,cursor = pg.return_postgres_cursor()
	desc_query = "select did, cid, term from snomed2.active_selected_descriptions"
	desc_df = pg.return_df_from_query(cursor, desc_query, None, ["did", "cid", "term"])

	desc_df['term'] = desc_df['term'].map(extract_value)

	engine = pg.return_sql_alchemy_engine()
	desc_df.to_sql('cleaned_selected_descriptions_prelim', engine, \
		schema='annotation2', if_exists='replace', index=False)

def active_cleaned_distinct_terms():
	conn,cursor = pg.return_postgres_cursor()
	engine = pg.return_sql_alchemy_engine()
	dist_query = "select distinct on (conceptid, term) \
		id, conceptid, term from annotation.active_cleaned_selected_concept_descriptions"
	dist_df = pg.return_df_from_query(cursor, dist_query, None, ["id", "conceptid", "term"])
	print("query complete")
	dist_df.to_sql('active_cleaned_selected_concept_descriptions', engine, \
		schema='annotation', if_exists='replace')


if __name__ == "__main__":
	active_cleaned_selected_concept_descriptions_prelim()