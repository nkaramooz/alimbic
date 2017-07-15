import pandas as pd
import re
import pickle
import sys
import psycopg2
from sqlalchemy import create_engine
from fuzzywuzzy import fuzz
from pglib import return_postgres_cursor, return_df_from_query, return_sql_alchemy_engine
import numpy as np
import time
import multiprocessing as mp


def return_query_snomed_annotation_v1(query):
	results_header = ['query', 'substring', 'substring_start_index', 'substring_end_index',
		'conceptid', 'concept_name', 'score']

	query = query.lower()
	column_names = ['conceptid', 'term']
	snomed_names = return_df_from_query("select * from snomed.metadata_concept_names", column_names)

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

		# evaluating need for searching sub-phrases of the query
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
	

	annotated_results = annotated_results.groupby(['query', 'substring', 'substring_start_index', 'substring_end_index', 'conceptid'], as_index=False)['score'].sum()

	if len(annotated_results) > 0:
		annotated_results['weighted_score'] = (annotated_results['substring_end_index'] - annotated_results['substring_start_index'])+1
	return annotated_results.reset_index(drop=True)

def return_query_snomed_annotation_v2(query):

	query = query.lower()

	query_words = query.split()

	filter_words = pd.read_pickle('filter_words')['words'].tolist()

	query_key_words = list(set(query_words) - set(filter_words))

	column_names = ['conceptid', 'term']

	sql_key_words_query = 'select distinct on (conceptid, term) conceptid, term from snomed.metadata_concept_key_words where ' + ' or ' .join(('word ilike \'' + str(n) + '\'' for n in query_key_words))

	snomed_names = return_df_from_query(sql_key_words_query, column_names)
	snomed_names_split_df = np.array_split(snomed_names, 4)

	pool = mp.Pool(processes=4)
	funclist = []

	for  df in snomed_names_split_df:
		funclist.append(pool.apply_async(snomed_annotation, (df, query)))

	pool.close()
	pool.join()
	results_df = pd.DataFrame()

	for r in funclist:
		results_df = results_df.append(r.get())

	results_df = results_df.groupby(['query', 'substring', 'substring_start_index', 'substring_end_index', 'conceptid'], as_index=False)['score'].sum()

	if len(results_df) > 0:
		results_df['weighted_score'] = (results_df['substring_end_index'] - results_df['substring_start_index'])+1
	return results_df

def snomed_annotation(snomed_names, query):
	results_header = ['query', 'substring', 'substring_start_index', 'substring_end_index',
		'conceptid', 'concept_name', 'score']

	query_words = query.split()

	annotated_results = pd.DataFrame(columns=results_header)

	for index,row in snomed_names.iterrows():
		snomed_term_len = len(row['term'].split())
		query_len = len(query_words)
		
		snomed_term = row['term'].decode('utf-8').lower()

		complete_score = fuzz.token_sort_ratio(query, snomed_term)
		if complete_score > 80:
			row_annotation = pd.DataFrame([[query, substring, 0.0, query_len-1, row['conceptid'], 
				snomed_term, complete_score]], columns=results_header)
			annotated_results = annotated_results.append(row_annotation)

		# evaluating need for searching sub-phrases of the query
		if query_len > snomed_term_len:
			
			counter = 0
			while counter + snomed_term_len <= query_len:

				substring = ' '.join(query_words[counter:counter+snomed_term_len])

				substring_score = fuzz.token_sort_ratio(substring, snomed_term)
				if substring_score > 80:
					row_annotation = pd.DataFrame([[query, substring, 
						counter, counter+snomed_term_len-1,
						row['conceptid'], snomed_term, substring_score]], columns=results_header)
					annotated_results = annotated_results.append(row_annotation)
				counter += 1

	return annotated_results

def prune_query_annotation(scores_df):
	exclude_index_arr = []
	scores_df = scores_df.sort_values('weighted_score', ascending=False)

	results_df = pd.DataFrame()
	results_index = []
	changes_made = False
	for index, row in scores_df.iterrows():
		if index not in exclude_index_arr:
			exclude = scores_df.index.isin(exclude_index_arr)
			subset_df = scores_df[~exclude].sort_values(['score', 'weighted_score'])
			subset_df = subset_df[(subset_df['substring_start_index'] >= row['substring_start_index'])
				& (subset_df['substring_end_index'] <= row['substring_end_index'])]
			subset_df = subset_df.sort_values(['score', 'weighted_score'], ascending=False)

			result = subset_df.iloc[0].copy()
			skip_top_match = False
			if len(subset_df) > 1:
				changes_made = True
				
				new_exclude = scores_df[(scores_df['substring_start_index'] >= result['substring_start_index']) &
					(scores_df['substring_end_index'] <= result['substring_end_index'])]
				exclude_index_arr.append(index)
				exclude_index_arr.extend(new_exclude.index.values)

				
			
			if len(result['substring'].split()) == 1:

				filter_words = pd.read_pickle('filter_words')
				
				for index,row in filter_words.iterrows():
					if fuzz.ratio(result['substring'], row['words']) > result['score']:
						skip_top_match = True
						break
			if not skip_top_match:
				results_df = results_df.append(result)
			
			# if result.name not in results_index:
			# 	results_index.append(result.name)
			# 	results_df = results_df.append(result)
			# exclude_index_arr.append(index)	

	if not changes_made:
		return results_df
	else:
		return prune_query_annotation(results_df)


def pprint(data_frame):
	with pd.option_context('display.max_rows', None, 'display.max_columns', 10):
		pd.set_option('display.width', 1000)
		print data_frame


if __name__ == "__main__":

	query = """
		chest pain and breath shortness at night congestive heart failure
	"""


	start_time = time.time()

	# scores = return_query_snomed_annotation_v2(query)
	# scores = scores.sort_values('weighted_score', ascending=False)
	# pprint(scores)
	# pruned = prune_query_annotation(scores)
	# pprint(pruned)
	print fuzz.token_sort_ratio('breath shortness', 'shortness of breath')
	# pprint(pruned)
	print("--- %s seconds ---" % (time.time() - start_time))
