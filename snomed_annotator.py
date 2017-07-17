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
from Levenshtein import *


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

def return_query_snomed_annotation_v3(query):

	filter_words_query = "select words from annotation.filter_words"
	filter_df = return_df_from_query(filter_words_query, ["words"])

	annotation_header = ['query', 'substring', 'substring_start_index', 'substring_end_index', 'conceptid']

	query = query.lower()
	query_words = query.split()

	candidate_df_arr = []
	results_df = pd.DataFrame()
	for index,word in enumerate(query_words):

		if (filter_df['words'] == word).any():
			continue
		else:
			## need an added step such that if you've gone enough words
			# and haven't matched outside the first word of a description
			# that the description drops off
			# that should speed things up a bit
			candidate_df_arr = evaluate_candidate_df(word, index, candidate_df_arr)

			candidate_df_arr, new_results_df = get_results(candidate_df_arr)
			results_df = results_df.append(new_results_df)

			new_candidate_query = "select description_id, conceptid, term, word, term_length, case when (1-levenshtein(word," + "\'" + word + "\')) < 0 then 0 else (1-levenshtein(word," + "\'" + word + "\')) end as l_dist from annotation.selected_concept_key_words where description_id in (select description_id from annotation.selected_concept_key_words where (1-levenshtein(word, " + "\'" + word + "\')) >=0.70) "
			new_candidate_df = return_df_from_query(new_candidate_query, ["description_id", "conceptid", "term", "word", "term_length", "l_dist"])
			

			new_candidate_df['substring_start_index'] = index
			candidate_df_arr.append(new_candidate_df)

		
	candidate_df_arr, new_results_df = get_results(candidate_df_arr)
	results_df = results_df.append(new_results_df)
	print results_df

def get_results(candidate_df_arr):
	results_df = pd.DataFrame()
	for index,df in enumerate(candidate_df_arr):
		#want description ids where none of the keywords have a 0 score
		# res_df = df.groupby(['conceptid', 'description_id'], as_index=False)['l_dist'].sum()
		# total_df = df.groupby(['conceptid', 'description_id'], as_index=False)['term_length'].sum()

		# find description_ids that should be excluded
		exclusion_series = df[df['l_dist'] == 0]['description_id'].tolist()
		new_results = df[~df['description_id'].isin(exclusion_series)]
		candidate_df_arr[index] = df[df['description_id'].isin(exclusion_series)]
		results_df = results_df.append(new_results)

	return candidate_df_arr,results_df


def evaluate_candidate_df(word, substring_start_index, candidate_df_arr):

	new_candidate_df_arr = []
	for index,df in enumerate(candidate_df_arr):

		df_copy = df.copy()
		for index,row in df_copy.iterrows():

			l_dist = fuzz.ratio(word, row['word'])

			if l_dist >= 80:
				df_copy.loc[index, 'l_dist'] = l_dist/100.00
				df_copy.loc[index, 'substring_start_index'] = substring_start_index

		new_candidate_df_arr.append(df_copy)

	for index, new_df in enumerate(new_candidate_df_arr):
		new_df_description_score = new_df.groupby(['description_id'], as_index=False)['l_dist'].sum()
		old_df_description_score = candidate_df_arr[index].groupby(['description_id'], as_index=False)['l_dist'].sum()

		if len(old_df_description_score) >= 1 and len(new_df_description_score) >= 1:
			candidate_descriptions = new_df_description_score[new_df_description_score['l_dist'] > old_df_description_score['l_dist']]
			new_candidate_df_arr[index] = new_df[new_df['description_id'].isin(candidate_descriptions['description_id'])]

	return new_candidate_df_arr


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

def apply_filter_words_exceptions():
	filter_words = pd.read_pickle('filter_words')
	filter_words = filter_words[
		(filter_words['words'] != 'disease') &
		(filter_words['words'] != 'skin') & 
		(filter_words['words'] != 'treatment') &
		(filter_words['words'] != 'symptoms') &
		(filter_words['words'] != 'syndrome') &
		(filter_words['words'] != 'pain')]
	filter_words.to_pickle('filter_words')

def update_postgres_filter_words():
	engine = return_sql_alchemy_engine()
	filter_words = pd.read_pickle('filter_words')
	filter_words.to_sql('filter_words', engine, schema='annotation', if_exists='replace')

if __name__ == "__main__":


	# for some reason adding "and nighttime cough breaks things"
	# might not be handling instances where result is 0 <- that is probably it
	query = """
		chest pain and congestive hart failure and nighttime cough
	"""

	
	# print ratio("breath", "breather")
	# apply_filter_words_exceptions()
	# update_postgres_filter_words()
	

	start_time = time.time()
	return_query_snomed_annotation_v3(query)
	# scores = return_query_snomed_annotation_v2(query)
	# scores = scores.sort_values('weighted_score', ascending=False)
	# pprint(scores)
	# pruned = prune_query_annotation(scores)
	# pprint(pruned)
	# print fuzz.token_sort_ratio('breath shortness', 'shortness of breath')
	# pprint(pruned)
	print("--- %s seconds ---" % (time.time() - start_time))
