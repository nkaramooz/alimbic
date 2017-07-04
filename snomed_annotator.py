import pandas as pd
import re
import pickle
import sys
import psycopg2
from fuzzywuzzy import fuzz
from pglib import return_postgres_cursor, return_df_from_query


def return_snomed_annotation(document_id, text):
	results_header = ['identifier', 'substring', 'substring_start_index', 'substring_end_index',
		'conceptid', 'concept_name', 'score']

	# for queries, trhe document_identifier is null
	# and the table prints the query as the "key"
	identifier = ""
	if document_id == None:
		identifier = text
		substring = text
	else:
		identifier = text
		substring = None

	text = text.lower()
	snomed_names = return_df_from_query("select * from snomed.metadata_concept_names")

	annotated_results = pd.DataFrame(columns=results_header)

	for index,row in snomed_names.iterrows():
		snomed_term_len = len(row['term'].split())
		text_len = len(text.split())
		snomed_term = row['term'].decode('utf-8').lower()

		complete_score = fuzz.ratio(text, snomed_term)
		if complete_score > 80:

			#want to store index of terms
			row_annotation = pd.DataFrame([[identifier, substring, 0, text_len-1, row['conceptid'], 
				snomed_term, complete_score]], columns=results_header)
			annotated_results = annotated_results.append(row_annotation)

		if text_len > snomed_term_len:
			counter = 0

			while counter + snomed_term_len <= text_len:
				word_array = text.split()[counter:counter+snomed_term_len]
				substring = ' '.join(word_array)

				#not dividing by length since for longer documents that would be bad
				#for queries we may want to do this.

				substring_score = fuzz.ratio(substring, snomed_term)
				if substring_score > 80:
					row_annotation = pd.DataFrame([[identifier, substring, counter, counter+snomed_term_len-1,
						row['conceptid'], snomed_term, substring_score]], columns=results_header)
					annotated_results = annotated_results.append(row_annotation)
				counter += 1
	return annotated_results.sort_values('score', ascending = False)




if __name__ == "__main__":
	query = "shortness of breath chest pain"
	print return_snomed_annotation(None, query)