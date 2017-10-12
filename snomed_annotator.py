import pandas as pd
import re
import pickle
import sys
import psycopg2
from sqlalchemy import create_engine
from fuzzywuzzy import fuzz
import pglib as pg
import numpy as np
import time
import multiprocessing as mp
import copy
import utils as u 


def get_new_candidate_df(word, cursor):

	#Knocks off a second off a sentence by selecting for word_ord=1
	new_candidate_query = "select description_id, conceptid, term, word, word_ord, term_length, case when word like %s \
		then 1 else 0 end as l_dist from annotation.selected_concept_key_words where \
		description_id in (select description_id from annotation.selected_concept_key_words where word like %s and word_ord=1)"
	new_candidate_df = pg.return_df_from_query(cursor, new_candidate_query, (word, word), \
	 ["description_id", "conceptid", "term", "word", "word_ord", "term_length", "l_dist"])
	return new_candidate_df


def return_line_snomed_annotation(cursor, line, threshold, filter_df):

	annotation_header = ['query', 'substring', 'substring_start_index', 'substring_end_index', 'conceptid']
	annotation_header = ['query', 'substring', 'substring_start_index', \
		'substring_end_index', 'conceptid']

	line = line.lower()
	ln_words = line.split()

	candidate_df_arr = []
	results_df = pd.DataFrame()

	for index,word in enumerate(ln_words):


		if (filter_df['words'] == word).any():
			continue
		else:

			candidate_df_arr = evaluate_candidate_df(word, index, candidate_df_arr, threshold)

			new_candidate_df = get_new_candidate_df(word, cursor)
			if len(new_candidate_df) > 0:
				new_candidate_df['substring_start_index'] = index
				new_candidate_df['description_start_index'] = index
				candidate_df_arr.append(new_candidate_df)
				candidate_df_arr, new_results_df = get_results(candidate_df_arr)
				results_df = results_df.append(new_results_df)
	
	if len(results_df) > 0:
		order_score = results_df

		order_score['order_score'] = (results_df['word_ord'] - (results_df['substring_start_index'] - \
			results_df['description_start_index'] + 1)).abs()
		order_score = order_score[['conceptid', 'description_id', 'description_start_index', 'order_score']].groupby(\
			['conceptid', 'description_id', 'description_start_index'], as_index=False)['order_score'].sum()
		

		distinct_results = results_df[['conceptid', 'description_id', 'description_start_index']].drop_duplicates()
		results_group = results_df.groupby(['conceptid', 'description_id', 'description_start_index'], as_index=False)

		sum_scores = results_group['l_dist'].mean().rename(columns={'l_dist' : 'sum_score'})
		sum_scores = sum_scores[sum_scores['sum_score'] >= (threshold/100.0)]

		start_indexes = results_group['substring_start_index'].min().rename(columns={'substring_start_index' : 'term_start_index'})
		end_indexes = results_group['substring_start_index'].max().rename(columns={'substring_start_index' : 'term_end_index'})

		joined_results = distinct_results.merge(sum_scores, on=['conceptid', 'description_id', 'description_start_index'])

		joined_results = joined_results.merge(start_indexes, on=['conceptid', 'description_id', 'description_start_index'])
		joined_results = joined_results.merge(end_indexes, on=['conceptid', 'description_id', 'description_start_index'])
		joined_results = joined_results.merge(order_score, on=['conceptid', 'description_id', 'description_start_index'])

		joined_results['final_score'] = joined_results['sum_score'] * np.where(joined_results['order_score'] > 0, 0.95, 1)
		joined_results['term_length'] = joined_results['term_end_index'] - joined_results['term_start_index'] + 1
		joined_results['final_score'] = joined_results['final_score'] + 0.04*joined_results['term_length']



		final_results = prune_results(joined_results)

		if len(final_results) > 0:
			final_results = add_names(final_results)
			return final_results

	return None

def get_results(candidate_df_arr):

	new_candidate_df_arr = []
	results_df = pd.DataFrame()
	for index,df in enumerate(candidate_df_arr):
		exclusion_series = df[df['l_dist'] == 0]['description_id'].tolist()
		new_results = df[~df['description_id'].isin(exclusion_series)]

		remaining_candidates = df[df['description_id'].isin(exclusion_series)]
		if len(remaining_candidates) != 0:
			new_candidate_df_arr.append(remaining_candidates)
		results_df = results_df.append(new_results)
		
	return new_candidate_df_arr,results_df


def evaluate_candidate_df(word, substring_start_index, candidate_df_arr, threshold):

	new_candidate_df_arr = []
	for index,df in enumerate(candidate_df_arr):

		df_copy = df.copy()
		for index,row in df_copy.iterrows():

			l_dist = fuzz.ratio(word, row['word'])

			# assign l_dist to only those that pass threshold
			if l_dist >= threshold: ### TUNE ME
				df_copy.loc[index, 'l_dist'] = l_dist/100.00
				df_copy.loc[index, 'substring_start_index'] = substring_start_index

		new_candidate_df_arr.append(df_copy)

	# now want to make sure that number has gone up from before
	# ideally should also at this point be pulling out complete matches.

	final_candidate_df_arr = []
	for index, new_df in enumerate(new_candidate_df_arr):
		new_df_description_score = new_df.groupby(['description_id'], as_index=False)['l_dist'].sum()
		old_df_description_score = candidate_df_arr[index].groupby(['description_id'], as_index=False)['l_dist'].sum()

		if len(old_df_description_score) > 0 and len(new_df_description_score) > 0:
			candidate_descriptions = new_df_description_score[new_df_description_score['l_dist'] > old_df_description_score['l_dist']]
			filtered_candidates = new_df[new_df['description_id'].isin(candidate_descriptions['description_id'])]
			if len(filtered_candidates) != 0:
				final_candidate_df_arr.append(filtered_candidates)

	return final_candidate_df_arr


def prune_results(scores_df):

	exclude_index_arr = []
	scores_df = scores_df.sort_values(['term_start_index'], ascending=True)

	results_df = pd.DataFrame()
	results_index = []
	changes_made = False
	for index, row in scores_df.iterrows():

		if index not in exclude_index_arr:

			exclude = scores_df.index.isin(exclude_index_arr)
			subset_df = scores_df[~exclude].sort_values(['final_score', 'term_length'])

			subset_df = subset_df[
  				((subset_df['term_start_index'] <= row['term_start_index']) 
  					& (subset_df['term_end_index'] >= row['term_start_index'])) 
  				| ((subset_df['term_start_index'] <= row['term_end_index']) 
  					& (subset_df['term_end_index'] >= row['term_end_index']))
  				| ((subset_df['term_start_index'] >= row['term_start_index'])
  					& ((subset_df['term_end_index'] <= row['term_end_index'])))]

			subset_df = subset_df.sort_values(['final_score', 'term_length'], ascending=False)

			result = subset_df.iloc[0].copy()
			if len(subset_df) > 1:
				changes_made = True
				
				new_exclude = subset_df

				exclude_index_arr.append(index)
				exclude_index_arr.extend(new_exclude.index.values)

			results_df = results_df.append(result)

	if not changes_made:
		return results_df
	else:
		return prune_results(results_df)

def add_names(results_df):
	cursor = pg.return_postgres_cursor()
	if results_df is None:
		return None
	else:
		search_query = "select distinct on (conceptid) conceptid, term from annotation.selected_concept_descriptions \
			where conceptid in %s"
		params = (tuple(results_df['conceptid']),)
		names_df = pg.return_df_from_query(cursor, search_query, params, ['conceptid', 'term'])
		results_df = results_df.merge(names_df, on='conceptid')
		return results_df

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

	# query = """
	# 	chronic obstructive pulmonary disease and congestive heart failure
	# """
	query1 = """
		protein c deficiency protein s deficiency
	"""
	query2 = """
		chronic obstructive pulmonary disease and congestive heart failure
	"""
	query3 = """
		Cough as night asthma congestion sputum
	"""
	query4 = """
		S/beta+-thalassemia
	"""
	check_timer = u.Timer("full")
	# pprint(add_names(return_query_snomed_annotation_v3(query, 87)))
	cursor = pg.return_postgres_cursor()
	# u.pprint(return_line_snomed_annotation(cursor, query1, 87))
	# u.pprint(return_line_snomed_annotation(cursor, query2, 87))
	# u.pprint(return_line_snomed_annotation(cursor, query3, 87))
	res = return_line_snomed_annotation(cursor, query4, 100)
	u.pprint(add_names(res))
	# print("--- %s seconds ---" % (time.time() - start_time))
	check_timer.stop()