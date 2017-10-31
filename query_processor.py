import pandas as pd
import re
import pickle
import sys
import psycopg2
from fuzzywuzzy import fuzz
from utilities.pglib import return_df_from_query



def return_filtered_query(text):
	filter_words = pd.read_pickle('filter_words')

	result_query = ""

	for word in text.split():
		word = word.lower()
		filter_words['fuzz_ratio'] = filter_words['words'].apply(lambda x: fuzz.ratio(x, word))

		if (filter_words['fuzz_ratio'] >= 80).any():
			continue
		else:
			result_query = result_query + word + ' '

	result_query = result_query.strip()
	return result_query

def annotate_query_with_snomed(query):
	query = query.lower()
	snomed_names = return_df_from_query("select * from snomed.distinct_concept_names limit 100")
	
	query_annotation = pd.DataFrame(columns=['query_terms', 'conceptid', 'concept_name', 'score'])

	for index,row in snomed_names.iterrows():
		ref_length = len(row['term'].split())
		query_length = len(query.split())
		snomed_term = row['term'].decode('utf-8').lower()

		if fuzz.ratio(query, snomed_term) > 40:
			row_annotation = pd.DataFrame([[query, row['conceptid'], snomed_term, fuzz.ratio(query, snomed_term)]], columns = ['query_terms', 'conceptid', 'concept_name', 'score'])
			query_annotation = query_annotation.append(row_annotation)

		if query_length > ref_length:
			counter = 0

			while counter + ref_length < query_length:
				word_array = query.split()[counter:counter+ref_length]
				substring = ' '.join(word_array)
				if fuzz.ratio(substring, snomed_term)/query_length > 40:
					row_annotation = pd.DataFrame([[substring, row['conceptid'], snomed_term, fuzz.ratio(substring, snomed_term)/(query_length)]], columns = ['query_terms', 'conceptid', 'concept_name', 'score'])
					query_annotation = query_annotation.append(row_annotation)
				counter += 1

	print query_annotation.sort_values('score', ascending = False)




if __name__ == "__main__":
	query = "COPD"
	print fuzz.ratio(query, "COPD - Chronic obstructive pulmonary disease")
		# annotate_query_with_snomed(query)
		# print return_filtered_query(query)