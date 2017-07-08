import pandas as pd
import re
import pickle
import sys
import psycopg2
from fuzzywuzzy import fuzz
from pglib import return_postgres_cursor, return_df_from_query
import numpy as np


def return_query_snomed_annotation(query):
	results_header = ['query', 'substring', 'substring_start_index', 'substring_end_index',
		'conceptid', 'concept_name', 'score']

	query = query.lower()
	snomed_names = return_df_from_query("select * from snomed.metadata_concept_names")

	annotated_results = pd.DataFrame(columns=results_header)

	for index,row in snomed_names.iterrows():
		snomed_term_len = len(row['term'].split())
		query_words = query.split()
		query_len = len(query_words)
		snomed_term = row['term'].decode('utf-8').lower()

		complete_score = fuzz.ratio(query, snomed_term)
		if complete_score > 80:

			row_annotation = pd.DataFrame([[query, substring, 0.0, query_len-1, row['conceptid'], 
				snomed_term, complete_score]], columns=results_header)
			annotated_results = annotated_results.append(row_annotation)

		if query_len > snomed_term_len:
			counter = 0

			while counter + snomed_term_len <= query_len:

				substring = ' '.join(query_words[counter:counter+snomed_term_len])

				substring_score = fuzz.ratio(substring, snomed_term)
				if substring_score > 80:
					row_annotation = pd.DataFrame([[query, substring, 
						counter, counter+snomed_term_len-1,
						row['conceptid'], snomed_term, substring_score]], columns=results_header)
					annotated_results = annotated_results.append(row_annotation)
				counter += 1

	# annotated_results = annotated_results.reset_index(drop=True)
	# return annotated_results.sort_values('score', ascending = False)


	# annotated_results['weighted_score'] = annotated_results['score'] * \
	# 			(1/(1+np.exp(0.005*((annotated_results['substring_end_index'] - annotated_results['substring_start_index']) + 1))))
	
	# annotated_results['weighted_score'] = annotated_results['score'] * \
	# 	1/(1+abs((annotated_results['substring_end_index'] - annotated_results['substring_start_index'])-3))
	
	pretty_print(annotated_results)
	annotated_results = annotated_results.groupby(['query', 'substring', 'substring_start_index', 'substring_end_index', 'conceptid'], as_index=False)['score'].sum()

	annotated_results['weighted_score'] = (annotated_results['substring_end_index'] - annotated_results['substring_start_index'])+1
	return annotated_results.reset_index(drop=True)


def prune_query_annotation(scores_df):

	results_df = pd.DataFrame()
	changes_made = False

	index_array = [] #it sucks that I have to do this, but can't figure out how else to proceed
	for index, row in scores_df.iterrows():

		exclude_df = scores_df.index.isin(index_array)

		subset_df = scores_df[~exclude_df].copy()

		subset_df = subset_df[
			((subset_df['substring_start_index'] <= row['substring_start_index']) 
			& (subset_df['substring_end_index'] >= row['substring_start_index'])) 
			| ((subset_df['substring_start_index'] <= row['substring_end_index']) 
			& (subset_df['substring_end_index'] >= row['substring_end_index']))]

		if len(subset_df) > 1:
			changes_made = True

			# subset_df = subset_df.sort_values('weighted_score', ascending = False)

			subset_df = subset_df.sort_values(['score', 'weighted_score'], ascending=False)

			top_match = subset_df.iloc[0].copy()

			skip_top_match = False
			if len(top_match['substring']) == 1:
				filter_words = pd.read_pickle('filter_words')

				
				for index,row in filter_words.iterrows():
					if fuzz.ratio(top_match['substring'], row['words']) > top_match['score']:
						skip_top_match = True
						break

			if not skip_top_match:
				results_df = results_df.append(subset_df.iloc[0].copy())

			index_array.extend(subset_df.index.values)
		elif len(subset_df) == 1:
			top_match = subset_df.iloc[0].copy()

			skip_top_match = False
			if len(top_match['substring']) == 1:
				filter_words = pd.read_pickle('filter_words')

				
				for index,row in filter_words.iterrows():
					if fuzz.ratio(top_match['substring'], row['words']) > top_match['score']:
						skip_top_match = True
						break

			if not skip_top_match:
				results_df = results_df.append(subset_df.iloc[0].copy())

			index_array.extend(subset_df.index.values)


	if changes_made:
		return prune_query_annotation(results_df)
	else:
		return results_df


def pretty_print(data_frame):
	with pd.option_context('display.max_rows', None, 'display.max_columns', 10):
		pd.set_option('display.width', 1000)
		print data_frame


if __name__ == "__main__":
	query = """
		The risk of sudden death has changed over time among patients with symptomatic heart failure and reduced ejection fraction with the sequential introduction of medications including angiotensin-converting-enzyme inhibitors, angiotensin-receptor blockers, beta-blockers, and mineralocorticoid-receptor antagonists. 
		We sought to examine this trend in detail.
	"""
	query = query.lower()
	snomed_names = return_df_from_query("select * from snomed.metadata_concept_names")

	snomed_names.to_pickle('snomed_names')
	# scores = return_query_snomed_annotation(query)

	# # pretty_print(scores)
	# pruned = prune_query_annotation(scores)
	# pretty_print(pruned)
	# filter_words = pd.read_pickle('filter_words')
	# pretty_print(filter_words)

	# new_df = pd.DataFrame([['a', 'b', 'c', 4], ['a', 'd', 'e', 2], ['d', 'g', 'l', 1]], columns=['A', 'B' ,'C','count'])
	# new_df = new_df.sort_values('count', ascending = False)

	# pretty_print(new_df)
	# pretty_print(new_df.groupby(['A', 'B'], as_index=False)['count'].sum())
	# index_array = []
	# for index, row in new_df.iterrows():

	# 	index_array.append(index)
	# 	bad_df = new_df.index.isin(index_array)
		
		
	# 	good_df = new_df[~bad_df]
	# 	index_array.append(good_df.index.values[0])
	# 	print index_array
	# 	print good_df
	# 	break
		

	# pretty_print(new_df)
	

	# print new_df.index.get_loc()
