import pandas as pd
import re
import sys
import psycopg2
from fuzzywuzzy import fuzz
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import nltk.data
import numpy as np

from sqlalchemy import create_engine
import copy
import utilities.pglib as pg
import utilities.utils2 as u
import unittest
import uuid
import time
import sys
import sqlite3

# Below is used to annotate abstracts; faster but no tolerance for typos
def get_new_candidate_df(word, case_sensitive, spellcheck_threshold):
	new_candidate_query = ""
	new_candidate_df = None
	conn, cursor = pg.return_postgres_cursor()

	if spellcheck_threshold != 100:
		if case_sensitive:
			new_candidate_query = """
				select 
					adid
					,acid
					,term
					,tb1.word
					,word_ord
					,term_length
					,is_acronym
				from annotation2.lemmas tb1
				join annotation2.first_word tb2
				  on tb1.adid = ANY(tb2.adid_agg)
				join (select unnest(%s) as query_word) tb3
				  on levenshtein(tb3.query_word, tb2.word) < 3
				where soundex(query_word) = soundex(tb2.word)
			"""
		else:
			for i,v in enumerate(word):
				word[i] = word[i].lower()

			new_candidate_query = """
				select 
					adid
					,acid
					,term
					,lower(tb1.word) as word
					,word_ord
					,term_length
					,is_acronym
				from annotation2.lemmas tb1
				join annotation2.first_word tb2
				  on tb1.adid = ANY(tb2.adid_agg)
				join (select unnest(%s) as query_word) tb3
				  on levenshtein(tb3.query_word, lower(tb2.word)) < 3
				where soundex(query_word) = soundex(tb2.word)
			"""
		new_candidate_df = pg.return_df_from_query(cursor, new_candidate_query, (list(word),), \
	 		["adid", "acid", "term", "word", "word_ord", "term_length", "is_acronym"])

	else:
		if case_sensitive:
			new_candidate_query = """
				select 
					adid
					,acid
					,term
					,tb1.word
					,word_ord
					,term_length
					,is_acronym
				from annotation2.lemmas tb1
				join annotation2.first_word tb2
				  on tb1.adid = ANY(tb2.adid_agg)
				where tb2.word in %s
			"""
		else:
			for i,v in enumerate(word):
				word[i] = word[i].lower()

			new_candidate_query = """
				select 
					adid
					,acid
					,term
					,lower(tb1.word) as word
					,word_ord
					,term_length
					,is_acronym
				from annotation2.lemmas tb1
				join annotation2.first_word tb2
				  on tb1.adid = ANY(tb2.adid_agg)
				where lower(tb2.word) in %s
			"""
		
		new_candidate_df = pg.return_df_from_query(cursor, new_candidate_query, (tuple(word),), \
		 ["adid", "acid", "term", "word", "word_ord", "term_length", "is_acronym"])

	cursor.close()
	conn.close()
	return new_candidate_df


def get_lemmas(ln_words, case_sensitive, check_pos, lmtzr):
	if check_pos:
		pos_tag = get_wordnet_pos(ln_words)
	
	if lmtzr is None:
		lmtzr = WordNetLemmatizer()

	ln_lemmas = []

	for i,w in enumerate(ln_words):

		if not case_sensitive:
			w = w.lower()
		else:
			if w.upper() != w:
				w = w.lower()

		if w.lower() != 'vs' and w.upper() != w:
			if w == 'doses':
				w = 'dose'
			elif w == 'does':
				w = 'do'
			elif w == 'induces' or w == 'induced' or w == 'inducing':
				w = 'cause'
			elif w == 'bound' or w == 'binding' or w == 'binds':
				w = 'bind'
			elif w == 'controlled' or w == 'controlling' or w == 'controls':
				w = 'control'
			elif w == 'has':
				w = 'have'
			elif w == 'cause' or w == 'causing' or w == 'causes' or w == 'caused':
				w = 'cause'
			elif w == 'computed' or w == 'computing' or w=='computes':
				w='compute'
			elif w == 'whiting':
				pass
			elif w=='pylori':
				pass
			elif w=='enrolling' or w=='enroll' or w=='enrolled' or w=='enrolls':
				w='enroll'
			elif w=='fasting':
				pass
			elif w=='ankylosing':
				pass
			elif w=='releasing' or w=='released' or w=='releases':
				w='release'
			elif w=='ovarian' or w=='ovaries':
				w='ovary'
			elif w=='opiate' or w=='opiates' or w=='opioids':
				w='opioid'
			elif w=='was' or w=='were' or w=='is':
				w='is'
			else:
				if check_pos:
					w = lmtzr.lemmatize(w, pos_tag[i])
				else:
					lmtzr.lemmatize('protein')
					w = lmtzr.lemmatize(w)
		ln_lemmas.append(w)
	return ln_lemmas


def get_wordnet_pos(ln_words):
	
	tags = nltk.pos_tag(ln_words)
	tag_dict = {"J" : wordnet.ADJ, "N" : wordnet.NOUN,
		"V": wordnet.VERB, "R" : wordnet.ADV, "I" : wordnet.VERB}

	res = []
	for i,tag in enumerate(tags):
		res.append(tag_dict.get(tag[1][0].upper(), wordnet.NOUN))
	return res


def return_line_snomed_annotation(line, spellcheck_threshold, case_sensitive, check_pos, cache, lmtzr):
	annotation_header = ['query', 'substring', 'substring_start_index', 'substring_end_index', 'acid', 'is_acronym']
	ln_words = line.split()
	ln_lemmas = get_lemmas(ln_words, case_sensitive, check_pos, lmtzr)
	
	if len(cache) == 0:
		words_df = pd.DataFrame()
		ln_words = line.split()
		for index,word in enumerate(ln_lemmas):
			words_df = words_df.append(pd.DataFrame([[index, word]], columns=['term_index', 'term']))
		words_df['line'] = line
		return words_df, None

	candidate_df_arr = []
	results_df = pd.DataFrame()
	words_df = pd.DataFrame()
	final_results = pd.DataFrame()

	# word is query word (not necessarily correctly spelled word)

	for index,word in enumerate(ln_lemmas):
		words_df = words_df.append(pd.DataFrame([[index, word]], columns=['term_index', 'term']))

		if len(candidate_df_arr) > 0:
			candidate_df_arr, active_match = evaluate_candidate_df_2(word, index, candidate_df_arr, spellcheck_threshold, case_sensitive)

		# the limitation here is that it won't catch instances where first and second word
		# of phrase are switched
		if spellcheck_threshold != 100:
			new_candidate_df = cache[(cache[word] >= spellcheck_threshold) & (cache.word_ord == 1)]
		else:
			new_candidate_df = cache[(cache['word'] == word) & (cache.word_ord == 1)]

		if len(new_candidate_df.index) > 0:
			if spellcheck_threshold != 100:
				new_candidate_ids = new_candidate_df['adid']
				new_candidate_df = cache[cache.adid.isin(new_candidate_ids)].copy()
				word_ord = new_candidate_df[new_candidate_df[word] >= spellcheck_threshold].groupby(['adid'], as_index=False)['word_ord'].min()
				word_ord['l_dist'] = 1.0
				new_candidate_df = new_candidate_df.merge(word_ord, on=['adid', 'word_ord'], how='left')
				new_candidate_df['l_dist'] = new_candidate_df['l_dist'].fillna(-1.0)
				new_candidate_df.loc[(new_candidate_df[word] >= spellcheck_threshold) & (new_candidate_df['l_dist'] == 1.0) \
					,'l_dist'] = new_candidate_df[word]
			else:
				new_candidate_ids = new_candidate_df['adid']
				new_candidate_df = cache[cache.adid.isin(new_candidate_ids)].copy()
				word_ord = new_candidate_df[new_candidate_df['word'] == word].groupby(['adid'], as_index=False)['word_ord'].min()
				word_ord['l_dist'] = 1.0
				new_candidate_df = new_candidate_df.merge(word_ord, on=['adid', 'word_ord'], how='left')
				new_candidate_df['l_dist'] = new_candidate_df['l_dist'].fillna(-1.0)
				new_candidate_df.loc[(new_candidate_df['word'] == word) & (new_candidate_df['l_dist'] == 1.0) \
					,'l_dist'] = 100

		if len(new_candidate_df.index) > 0:
			new_candidate_df['substring_start_index'] = index
			new_candidate_df['description_start_index'] = index
			candidate_df_arr.append(new_candidate_df)

		candidate_df_arr, new_results_df = get_results(candidate_df_arr, index)
		results_df = results_df.append(new_results_df, sort=False)

	if len(results_df.index) > 0:
		order_score = results_df

		order_score['order_score'] = (results_df['word_ord'] - (results_df['substring_start_index'] - \
			results_df['description_start_index'] + 1)).abs()

		order_score = order_score[['acid', 'adid', 'description_start_index', 'description_end_index', 'term', 'order_score']].groupby(\
			['acid', 'adid', 'description_start_index'], as_index=False)['order_score'].sum()
		
		distinct_results = results_df[['acid', 'adid', 'description_start_index', 'description_end_index', 'term']].drop_duplicates()
	
		results_group = results_df.groupby(['acid', 'adid', 'description_start_index'], as_index=False)

		sum_scores = results_group['l_dist'].mean().rename(columns={'l_dist' : 'sum_score'})
		sum_scores = sum_scores[sum_scores['sum_score'] >= spellcheck_threshold]

		start_indexes = results_group['substring_start_index'].min().rename(columns={'substring_start_index' : 'term_start_index'})
		end_indexes = results_group['substring_start_index'].max().rename(columns={'substring_start_index' : 'term_end_index'})

		joined_results = distinct_results.merge(sum_scores, on=['acid', 'adid', 'description_start_index'])

		joined_results = joined_results.merge(start_indexes, on=['acid', 'adid', 'description_start_index'])
		joined_results = joined_results.merge(end_indexes, on=['acid', 'adid', 'description_start_index'])
		joined_results = joined_results.merge(order_score, on=['acid', 'adid', 'description_start_index'])
		
		joined_results['final_score'] = joined_results['sum_score'] * np.where(joined_results['order_score'] > 0, 0.95, 1)
		joined_results['term_length'] = joined_results['term_end_index'] - joined_results['term_start_index'] + 1
		joined_results['final_score'] = joined_results['final_score'] + 50*joined_results['term_length']
	
		final_results = prune_results_v2(joined_results, joined_results)
	
		final_results = final_results.merge(results_df, on=['adid'])

		final_results = final_results[['acid_x', 'adid', 'description_start_index_x', 'description_end_index_x', 'term_x', 'sum_score', 'term_start_index', \
			'term_end_index', 'order_score_x', 'final_score', 'term_length_x', 'is_acronym']]
		final_results.columns = ['acid', 'adid', 'description_start_index', 'description_end_index', 'term', 'sum_score', 'term_start_index', \
			'term_end_index', 'order_score', 'final_score', 'term_length', 'is_acronym']
	
	words_df['line'] = line

	return words_df, final_results


def get_results(candidate_df_arr, end_index):
	## no description here
	new_candidate_df_arr = []
	results_df = pd.DataFrame()

	for index,df in enumerate(candidate_df_arr):
		exclusion_series = df[df['l_dist'] == -1.0]['adid'].tolist()
		new_results = df[~df['adid'].isin(exclusion_series)].copy()
		new_results['description_end_index'] = end_index
		
		remaining_candidates = df[df['adid'].isin(exclusion_series)]
		if len(remaining_candidates.index) > 0:
			new_candidate_df_arr.append(remaining_candidates)
		results_df = results_df.append(new_results, sort=False)

	return new_candidate_df_arr,results_df

# May need to add exceptions for approved acronyms / common acronyms that don't need support
def acronym_check(results_df):
	non_acronyms_df = results_df[results_df['is_acronym'] == 0].copy()
	
	cid_counts = non_acronyms_df['acid'].value_counts()
	cid_cnt_df = pd.DataFrame({'acid':cid_counts.index, 'count':cid_counts.values})

	acronym_df = results_df[results_df['is_acronym'] == True].copy()
	acronym_df = acronym_df.merge(cid_cnt_df, on=['acid'],how='left')
	approved_acronyms = acronym_df[acronym_df['count'] >= 1]
	final = non_acronyms_df.append(approved_acronyms, sort=False)
	return final

	
# TODO: Leutinizing hormone releasing hormone messes up because hormone appears twice
def evaluate_candidate_df(word, substring_start_index, candidate_df_arr, threshold, case_sensitive):
	threshold = threshold/100.0
	new_candidate_df_arr = []
	for index,df in enumerate(candidate_df_arr):
		df_copy = df.copy()

		if case_sensitive:
			df_copy['l_dist_tmp'] = df_copy['word'].apply(lambda x: fuzz.ratio(x, word)/100.0)

		else:
			df_copy['l_dist_tmp'] = df_copy['word'].apply(lambda x: fuzz.ratio(x.lower(), word.lower())/100.0)

		df_copy.loc[(df_copy.l_dist == -1.0) & (df_copy.l_dist_tmp >= threshold) & \
			(df_copy.l_dist < threshold), ['substring_start_index']] = substring_start_index
		df_copy.loc[(df_copy.l_dist == -1.0) & (df_copy.l_dist_tmp >= threshold) & \
			(df_copy.l_dist < threshold), ['l_dist']] = df_copy[(df_copy.l_dist == -1.0) & \
			(df_copy.l_dist_tmp >= threshold)][['l_dist_tmp']].values
		df_copy = df_copy.drop(['l_dist_tmp'], axis=1)
	
		new_candidate_df_arr.append(df_copy)


	# now want to make sure that number has gone up from before
	# ideally should also at this point be pulling out complete matches.

	final_candidate_df_arr = []
	active_match = False

	for index, new_df in enumerate(new_candidate_df_arr):
		new_df_description_score = new_df.groupby(['adid'], as_index=False)['l_dist'].sum()
		old_df_description_score = candidate_df_arr[index].groupby(['adid'], as_index=False)['l_dist'].sum()



		if len(old_df_description_score.index) > 0 and len(new_df_description_score.index) > 0:
			candidate_descriptions = new_df_description_score[new_df_description_score['l_dist'] > old_df_description_score['l_dist']]
			filtered_candidates = new_df[new_df['adid'].isin(candidate_descriptions['adid'])]
			if len(filtered_candidates.index) != 0:
				final_candidate_df_arr.append(filtered_candidates)
				active_match = True

	return final_candidate_df_arr, active_match

def evaluate_candidate_df_2(word, substring_start_index, candidate_df_arr, spellcheck_threshold, case_sensitive):
	new_candidate_df_arr = []
	for index,df in enumerate(candidate_df_arr):
		df_copy = df.copy()
		
		# word_ord = df_copy[df_copy[word] >= spellcheck_threshold].groupby(['adid'], as_index=False)['word_ord'].min()
		# 	word_ord['l_dist'] = 1.0
		# 	new_candidate_df = new_candidate_df.merge(word_ord, on=['adid', 'word_ord'], how='left')
		# 	new_candidate_df['l_dist'] = new_candidate_df['l_dist'].fillna(-1.0)
		# 	new_candidate_df.loc[(new_candidate_df[word] >= spellcheck_threshold) & (new_candidate_df['l_dist'] == 1.0) \
		# 		,'l_dist'] = new_candidate_df[word]

		if spellcheck_threshold != 100:
			word_ord = df_copy[(df_copy[word] >= spellcheck_threshold) & (df_copy['l_dist'] == -1.0)].groupby(['adid'], as_index=False)['word_ord'].min()
			word_ord['tmp'] = 1.0
			df_copy = df_copy.merge(word_ord, on=['adid', 'word_ord'], how='left')
			df_copy.loc[(df_copy.l_dist == -1.0) & (df_copy[word] >= spellcheck_threshold)
				& (df_copy['tmp'] == 1.0), \
				['substring_start_index']] = substring_start_index
			df_copy.loc[(df_copy.l_dist == -1.0) & (df_copy[word] >= spellcheck_threshold) & (df_copy['tmp'] == 1.0), \
				['l_dist']] = df_copy[(df_copy.l_dist == -1.0) & \
				(df_copy[word] >= spellcheck_threshold) & (df_copy['tmp'] == 1.0)][[word]].values
			df_copy = df_copy.drop(columns=['tmp'])
			new_candidate_df_arr.append(df_copy)
		else:
			word_ord = df_copy[(df_copy['word'] == word) & (df_copy['l_dist'] == -1.0)].groupby(['adid'], as_index=False)['word_ord'].min()
			word_ord['tmp'] = 1.0
			df_copy = df_copy.merge(word_ord, on=['adid', 'word_ord'], how='left')
			df_copy.loc[(df_copy.l_dist == -1.0) & (df_copy['word'] == word)
				& (df_copy['tmp'] == 1.0), \
				['substring_start_index']] = substring_start_index
			df_copy.loc[(df_copy.l_dist == -1.0) & (df_copy['word'] == word) & (df_copy['tmp'] == 1.0), \
				['l_dist']] = 100
			df_copy = df_copy.drop(columns=['tmp'])
			new_candidate_df_arr.append(df_copy)

	# now want to make sure that number has gone up from before
	# ideally should also at this point be pulling out complete matches.

	final_candidate_df_arr = []
	active_match = False

	for index, new_df in enumerate(new_candidate_df_arr):
		new_df_description_score = new_df.groupby(['adid'], as_index=False)['l_dist'].sum()
		old_df_description_score = candidate_df_arr[index].groupby(['adid'], as_index=False)['l_dist'].sum()

		if len(old_df_description_score.index) > 0 and len(new_df_description_score.index) > 0:
			candidate_descriptions = new_df_description_score[new_df_description_score['l_dist'] > old_df_description_score['l_dist']]
			filtered_candidates = new_df[new_df['adid'].isin(candidate_descriptions['adid'])]
			if len(filtered_candidates.index) != 0:
				final_candidate_df_arr.append(filtered_candidates)
				active_match = True

	return final_candidate_df_arr, active_match


def prune_results_v2(scores_df, og_results):

	exclude_index_arr = []
	scores_df = scores_df.sort_values(['term_start_index'], ascending=True)

	results_df = pd.DataFrame()
	results_index = []
	changes_made = False

	scores_dict = scores_df.to_dict('index')

	for index in scores_dict:
		row = scores_dict[index]

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
			if len(subset_df.index) > 1:
				changes_made = True
				
				new_exclude = subset_df

				exclude_index_arr.append(index)
				exclude_index_arr.extend(new_exclude.index.values)

			results_df = results_df.append(result, sort=False)

	if not changes_made:
		# final_results = pd.DataFrame()

		final_results = og_results.merge(results_df[['term_start_index','term_end_index']], how='inner', left_on=['term_start_index','term_end_index'], right_on=['term_start_index','term_end_index']).copy()
		final_results.rename(columns={'term_start_index_x' : 'term_start_index', 'term_end_index_x' : 'term_end_index'})
	
		# for index,row in results_df.iterrows():

		# 	expanded_res = og_results[ \
		# 		(og_results['term_start_index'] == row['term_start_index']) \
		# 		& (og_results['term_end_index'] == row['term_end_index'])]
			
		# 	final_results = final_results.append(expanded_res, sort=False)

		return final_results
	else:
		return prune_results_v2(results_df, og_results)

# Attempts to disambiguate terms to select one
# Technically currently in the first IF statement if there is still a tie, it should resort
# to using previous concept frequencies. Examples where this would pertain is where a 
# document references pneumocystis pneumonia and phencyclidine, both of which have the
# acronym PCP
# will defer fixing this till later
def resolve_conflicts(results_df):
	acronym_df = results_df[results_df['is_acronym'] == True].copy()
	results_df = results_df[results_df['is_acronym'] == False].copy()

	counts_df = results_df.drop_duplicates(['term_start_index','acid', 'section_ind', 'ln_num']).copy()
	counts_df['index_count'] = 1
	counts_df = counts_df.groupby(['term_start_index', 'section_ind', 'ln_num'], as_index=False)['index_count'].sum()

	conflicted_indices = counts_df[counts_df['index_count'] > 1]

	conflict_free_df = results_df[~((results_df['term_start_index'].isin(conflicted_indices['term_start_index'])) & \
		(results_df['ln_num'].isin(conflicted_indices['ln_num'])) &\
		(results_df['section_ind'].isin(conflicted_indices['section_ind'])))].copy()

	conflicted_df = results_df[(results_df['term_start_index'].isin(conflicted_indices['term_start_index'])) & \
		(results_df['ln_num'].isin(conflicted_indices['ln_num'])) & \
		(results_df['section_ind'].isin(conflicted_indices['section_ind']))].copy()
	final_results = conflict_free_df.copy()


	if len(conflict_free_df.index) > 0:
		conflict_free_df['concept_count'] = 1
		concept_weights = conflict_free_df.groupby(['acid'], as_index=False)['concept_count'].sum()

		join_weights = conflicted_df.merge(concept_weights, on=['acid'],how='left')

		#set below to negative 10 and drop all rows with negative value
		join_weights['concept_count'].fillna(0, inplace=True)
		join_weights['final_score'] = join_weights['final_score'] + join_weights['concept_count']

		# Reason to sort by acid is to have something slightly deterministic in the event both scores are the same
		join_weights = join_weights.sort_values(['section_ind', 'ln_num', 'term_start_index', 'final_score', 'acid'], ascending=False)
		last_term_start_index = None
		last_ln_num = None
		last_section_ind = None

		for index,row in join_weights.iterrows():
			if last_term_start_index != row['term_start_index'] or last_ln_num != row['ln_num'] or \
				last_section_ind != row['section_ind']:
				final_results = final_results.append(row, sort=False)
				last_term_start_index = row['term_start_index']
				last_ln_num = row['ln_num']
				last_section_ind = row['section_ind']

	# Really should only want below for query annotation. Not document annotation
	 #first try and choose most common concept. If not choose randomly
	# else:
	## ADD BACK CURSOR HERE
	# 	u.p(results_df)
	# 	u.pprint(final_results)
	# 	u.pprint("STOP")
	# 	sys.exit(0)
		# conc_count_query = "select acid, count from annotation.concept_counts where acid in %s"
		# params = (tuple(conflicted_df['acid']),)
		# cid_cnt_df = pg.return_df_from_query(cursor, conc_count_query, params, ['acid', 'cnt'])


		# conflicted_df = conflicted_df.merge(cid_cnt_df, on=['acid'], how='left')
		# conflicted_df['cnt'] = conflicted_df['cnt'].fillna(value=1)

		# conflicted_df['final_score'] = conflicted_df['final_score'] * conflicted_df['cnt']

		# conflicted_df = conflicted_df.sort_values(['section_ind', 'ln_num', 'term_start_index', 'final_score'], ascending=False)

		# last_term_start_index = None
		# last_ln_num = None
		# last_section_ind = None
		# for index,row in conflicted_df.iterrows():
		# 	if last_term_start_index != row['term_start_index'] or last_ln_num != row['ln_num'] or \
		# 		last_section_ind != row['last_section_ind']:
		# 		final_results = final_results.append(row, sort=False)
		# 		last_term_start_index = row['term_start_index']
		# 		last_ln_num = row['ln_num']
		# 		last_section_ind = row['section_ind']

	final_results = final_results.append(acronym_df)

	return final_results


def add_names(results_df):
	conn,cursor = pg.return_postgres_cursor()
	if results_df is None:
		cursor.close()
		return None
	else:
		# Using old table since augmented tables include the acronyms
		search_query = "select acid, term from annotation2.preferred_concept_names \
			where acid in %s"

		params = (tuple(results_df['acid']),)
		names_df = pg.return_df_from_query(cursor, search_query, params, ['acid', 'term'])

		results_df = results_df.merge(names_df, on='acid')
		cursor.close()
		conn.close
		return results_df

def add_names_2(results_df):
	conn,cursor = pg.return_postgres_cursor()
	if results_df is None:
		cursor.close()
		return None
	else:
		# Using old table since augmented tables include the acronyms
		search_query = "select acid as parent_acid, term from annotation2.preferred_concept_names \
			where acid in %s"

		params = (tuple(results_df['parent_acid']),)
		names_df = pg.return_df_from_query(cursor, search_query, params, ['parent_acid', 'term'])

		results_df = results_df.merge(names_df, on='parent_acid')
		cursor.close()
		conn.close
		return results_df


def annotate_line(sentence_df, case_sensitive, check_pos, cache, \
		spellcheck_threshold, lmtzr):	

	line = clean_text(sentence_df['line'])

	words_df, annotation = return_line_snomed_annotation(line=line, spellcheck_threshold=spellcheck_threshold,\
			case_sensitive=case_sensitive, check_pos=check_pos, cache=cache, lmtzr=lmtzr)

	if annotation is not None:
		annotation['ln_num'] = sentence_df['ln_num']
		annotation['section'] = sentence_df['section']
		annotation['section_ind'] = sentence_df['section_ind']
		words_df['ln_num'] = sentence_df['ln_num']
		words_df['section'] = sentence_df['section']
		words_df['section_ind'] = sentence_df['section_ind']
	else:
		words_df['ln_num'] = 0
		words_df['section'] = sentence_df['section']
		words_df['section_ind'] = sentence_df['section_ind']

	return words_df, annotation

def get_annotated_tuple(c_df):
	if len(c_df) >= 1:
		c_df = c_df.sort_values(by=['term_index'], ascending=True)
		c_res = pd.DataFrame()
		sentence_arr = []
		line = c_df['line'].values[0]

		ln_words = line.split()
		concept_counter = 0
		concept_len = len(c_df)\

		c_dict = c_df.to_dict('records')
		for row in c_dict:
			if isinstance(row['acid'], str):
				sentence_arr.append((row['term'], row['acid']))
			else:
				sentence_arr.append((row['term'], 0))

		# for index,item in c_df.iterrows():
		# 	if isinstance(item['acid'], str):
		# 		sentence_arr.append((item['term'], item['acid']))
		# 	else:
		# 		sentence_arr.append((item['term'], 0))

		return sentence_arr
	else:
		return None


# when child_candidates is provided, it does not expand the query
# but rather reformats arrays around primary cids and filters
# to only child acids that exist in the search results
def query_expansion(conceptid_series, child_candidates, cursor):
	conceptid_tup = tuple(conceptid_series)

	if child_candidates is None and len(conceptid_tup) > 0:

		## artificially need to limit query size to not overwhelm elastic search
		child_query = """
			select
				child_acid
				,parent_acid
			from snomed2.transitive_closure_acid where parent_acid in %s
		"""
		# get rid of limit 30 when come up with more clever way

		child_df = pg.return_df_from_query(cursor, child_query, (conceptid_tup,), \
			["child_acid", "parent_acid"])

		
	elif child_candidates is not None and len(conceptid_tup) > 0:
		child_candidates_tup = tuple(child_candidates)
		child_query = """
			select
				child_acid
				,parent_acid
			from snomed2.transitive_closure_acid where parent_acid in %s
			and child_acid in %s
		"""

		child_df = pg.return_df_from_query(cursor, child_query, (conceptid_tup, child_candidates_tup), \
			["child_acid", "parent_acid"])
		
	else:
		return []

	results_list = []
	for item in conceptid_series:
		temp_res = [item]
		added_other = False
		# JUST CHANGED PARENTHESES location

		if len(child_df[child_df['parent_acid'] == item].index) > 0:
			temp_res.extend(child_df[child_df['parent_acid'] == item]['child_acid'].tolist())
			added_other = True

		if added_other:
			results_list.append(temp_res)
		else:
			results_list.extend(temp_res)

	return results_list

def get_children(conceptid, cursor):
	child_query = """
		select 
			child_acid
		from snomed2.transitive_closure_acid tb1
		left join annotation2.concept_counts tb2
			on tb1.child_acid = tb2.concept
		where tb1.parent_acid = %s and tb2.cnt is not null
		order by tb2.cnt desc
		limit 15
	"""
	child_df = pg.return_df_from_query(cursor, child_query, (conceptid,), \
		["child_acid"])
	

	return child_df['child_acid'].tolist()

def get_all_words_list(text, lmtzr):
	tokenized = nltk.sent_tokenize(text)
	all_words = []
	if lmtzr is None:
		lmtzr = WordNetLemmatizer()
	for ln_num, line in enumerate(tokenized):
		words = line.split()

		for index,w in enumerate(words):

			if w not in all_words and w.strip() != '':
				all_words.append(w)

	return all_words

def get_cache(all_words_list, case_sensitive, check_pos, spellcheck_threshold, lmtzr):
	lemmas = get_lemmas(all_words_list, case_sensitive, check_pos, lmtzr)
	cache = get_new_candidate_df(lemmas, case_sensitive, spellcheck_threshold)

	if len(cache) > 0:
		if spellcheck_threshold != 100:
			cache = cache.to_dict('records')
			for row in cache:
				max_fuzz = 0
				for i in lemmas:
					new_fuzz = fuzz.ratio(i, row['word'])
					if new_fuzz > max_fuzz:
						max_fuzz = new_fuzz
					row[i] = new_fuzz
				row['max_fuzz'] = max_fuzz
			cache = pd.DataFrame.from_dict(cache)

			cache = cache[cache['max_fuzz'] >= spellcheck_threshold]
			cache['points'] = 1
		else:
			cache['points'] = 0
			cache.loc[cache.word.isin(lemmas), 'points'] = 1

		csf = cache[['adid', 'term_length', 'points']].groupby(['adid', 'term_length'], as_index=False)['points'].sum()

		candidate_dids = csf[csf['term_length'] == csf['points']]['adid'].tolist()
		cache = cache[cache['adid'].isin(candidate_dids)].copy()
		cache = cache.drop(['points'], axis=1)
		return cache
	else:
		return pd.DataFrame()


# Returns 3 dataframes to be eventually materialized into tables
# sentence_annotations_df colums=['id (sentnece)', 'section', 'section_ind', 'ln_num', 'acid', 'adid', 'final_ann']
# ann_df returned for annotation if write sentences is negative
# otherwise returns 3 dataframes for pubmed indexing
# This is probably poor form


def annotate_text_not_parallel(sentences_df, cache, case_sensitive, \
	check_pos, bool_acr_check, write_sentences, lmtzr, spellcheck_threshold):
	
	concepts_df = pd.DataFrame()
	sentence_df = pd.DataFrame()
	words_df = pd.DataFrame()

	sentence_dict = sentences_df.to_dict('records')

	for row in sentence_dict:
		line_words_df, res_df = annotate_line(sentence_df=row, case_sensitive=case_sensitive, \
			check_pos=check_pos, cache=cache, spellcheck_threshold=spellcheck_threshold, lmtzr=lmtzr)

		if res_df is not None:
			concepts_df = concepts_df.append(res_df, sort=False)
		words_df = words_df.append(line_words_df, sort=False)

	if len(concepts_df.index) > 0:
		concepts_df = resolve_conflicts(concepts_df)

		if bool_acr_check:
			concepts_df = acronym_check(concepts_df)

	concepts_df = concepts_df.drop_duplicates(subset=['acid', 'description_start_index', 'section_ind', 'ln_num']).copy()

	conn = sqlite3.connect(':memory:')

	if len(concepts_df.index) > 0:
		words_df.to_sql('words_df', conn, index=False)
		concepts_df.to_sql('final_results', conn, index=False, if_exists='replace')
		query = """
			select 
				t1.term
				,t1.term_index
				,t1.ln_num
				,t1.section_ind
				,t1.section
				,t2.description_start_index
				,t2.acid
				,t2.adid
				,t2.description_end_index
				,t2.sum_score
				,t2.term_start_index
				,t2.term_end_index
				,t2.order_score
				,t2.final_score
				,t2.term_length
				,t2.is_acronym
				,t1.line
			from words_df t1
			left join final_results t2
			on t1.term_index >= t2.description_start_index and t1.term_index <= t2.description_end_index
				and t1.ln_num = t2.ln_num and t1.section_ind = t2.section_ind
		"""
		ann_df = pd.read_sql_query(query, conn)
	# Scenario where only unmatched terms in sentence
	else:
		ann_df = words_df.copy()
		ann_df['acid'] = np.nan
		ann_df['adid'] = np.nan

		ann_df.insert(0, 'description_start_index', 0, 0+len(ann_df))
		ann_df.insert(0, 'description_end_index', 0, 0+len(ann_df))
		# ann_df['description_start_index'] = words_df['term_index']
		# ann_df['description_end_index'] = words_df['term_index']


	section_max = ann_df['section_ind'].max()+1
	sentence_annotations_df = pd.DataFrame()
	sentence_tuples_df = pd.DataFrame()
	sentence_concept_arr_df = pd.DataFrame()
	if write_sentences:	
		for i in range(section_max):
			section_df = ann_df[ann_df['section_ind'] == i].copy()

			if len(section_df.index) > 0:
				ln_len = int(section_df['ln_num'].max())+1

				for j in range(ln_len):
					ln_df =  section_df[section_df['ln_num'] == j].copy()

					if len(ln_df.index) > 0:
						concept_arr = list(set(ln_df['acid'].dropna().tolist()))
						uid = str(uuid.uuid1())
					
						s_arr = get_annotated_tuple(ln_df)

						ln_df['sentence_id'] = uid
						ln_df.fillna('-1', inplace=True)
						
						ln_df.to_sql('ln_df', conn, index=False, if_exists='replace')
						query = """
							select 
								sentence_id
								,section
								,section_ind
								,ln_num
								,acid
								,adid
								,case when acid='-1' then term else acid end as final_ann
							from ln_df
							group by sentence_id,section,section_ind,ln_num, description_start_index, description_end_index, final_ann
						"""
						sentence_annotations_df = sentence_annotations_df.append(pd.read_sql_query(query,conn), sort=False)

						single_ln_df = ln_df[['sentence_id', 'section', 'section_ind', 'ln_num']].copy()
						
						try: 
							single_ln_df = single_ln_df.iloc[[0]].copy()
						except:
							print("ERROR ERROR")
							print(single_ln_df)
							u.pprint(ann_df)
						single_ln_df['sentence_tuples'] = [s_arr]
						sentence_tuples_df = sentence_tuples_df.append(single_ln_df, sort=False)
						single_concept_arr_df = single_ln_df[['sentence_id', 'section', 'section_ind', 'ln_num']].copy()
						single_concept_arr_df['concept_arr'] = [concept_arr]
						sentence_concept_arr_df = sentence_concept_arr_df.append(single_concept_arr_df, sort=False)
		return sentence_annotations_df, sentence_tuples_df, sentence_concept_arr_df

	return ann_df



### UTILITY FUNCTIONS

def clean_text(line):
	line = line.replace('.(', ' ')
	line = line.replace('.', '')
	line = line.replace('!', '')
	line = line.replace(',', '')
	line = line.replace(';', '')
	line = line.replace('*', '')
	line = line.replace('[', ' ')
	line = line.replace(']', ' ')
	line = line.replace('-', ' ')
	line = line.replace(':', ' ')
	line = line.replace('\'', '')
	line = line.replace('"', '')
	line = line.replace(':', '')
	line = line.replace('(', '')
	line = line.replace(')', '')
	line = line.replace('\\', ' ')
	line = line.replace('/', ' ')
	line = line.replace('  ', ' ')
	# This makes you lose the likelihood ratios
	# line = re.sub("\(.*?\)","",line)
	return line

def timeLimit(t, ref):
	return (t <= ref)

class TestAnnotator(unittest.TestCase):

	def test_equal(self):
		queryArr = [
			("protein C deficiency protein S deficiency", 100, 'complete', ['601882', '278172'])
			,("chronic obstructive pulmonary disease and congestive heart failure", 100, 'partial', ['55000', '154571'])
			,("chronic obstructive pulmonary disease and congestive heart failure", 80,'partial', ['55000', '154571'])
			,("protein C deficiency protein S deficiency", 80,'complete', ['601882', '278172'])
			,("ankyalosing spondylitis", 80,'complete', ['103044'])
			,("Weekly vs. Every-3-Week Paclitaxel and Carboplatin for Ovarian Cancer", 100, 'partial', ['172626', '237562', '105415'])
			,("Weekly vs. Every-3-Week Paclitaxel and Carboplatin for Ovarian Cancier", 80, 'partial', ['172626', '237562', '105415'])
			,("Sustained recovery of progressive multifocal leukoencephalopathy after treatment with IL-2.", 100, 'partial', ['73561', '197836'])
			,("luteinizing hormone releasing hormone in thyroid hormone deficiency", 100, 'partial', ['11008', '111861'])
			]

		lmtzr = WordNetLemmatizer()
		lmtzr.lemmatize("cough")
		

		u.pprint("============================")
		for q in queryArr:
			print(q)
			term = q[0]
			term = clean_text(term)
			spellcheck_threshold = q[1]
			all_words = get_all_words_list(term, lmtzr)

			s = u.Timer('get_cache')
			cache = get_cache(all_words_list=all_words, case_sensitive=True, \
				check_pos=False, spellcheck_threshold=spellcheck_threshold, lmtzr=lmtzr)
			sentences_df = pd.DataFrame([[term, 'title', 0,0]], \
				columns=['line', 'section', 'section_ind', 'ln_num'])
			item = pd.DataFrame([[term, 'title', 0, 0]], columns=['line', 'section', 'section_ind', 'ln_num'])

			res, g, s = annotate_text_not_parallel(sentences_df=sentences_df, cache=cache, \
				case_sensitive=True, check_pos=False, bool_acr_check=False, \
				spellcheck_threshold=spellcheck_threshold, \
				write_sentences=True, lmtzr=None)

			if q[2] == 'complete':
				self.assertTrue(set(res['acid'].values) == set(q[3]))
			else:
				self.assertTrue(set.intersection(set(q[3]), set(res['acid'].values))==set(q[3]))

		u.pprint("============================")


if __name__ == "__main__":

	query1 = "protein C deficiency protein S deficiency"
	query2 = "chronic obstructive pulmonary disease and congestive heart failure"
	query3 = "Cough at night asthma congestion sputum"
	query4 = "H2-receptor antagonists"
	query5 = "Heart Failure with Preserved Ejection Fraction."
	query7 = "Weekly vs. Every-3-Week Paclitaxel and Carboplatin for Ovarian Cancer"
	query8 = "Diastolic heart failure--abnormalities in active relaxation and passive stiffness of the left ventricle."
	query9 = "Grave's disease"
	query10 = "exacerbation of chronic obstructive pulmonary disease"
	query11= "cancer of ovary"
	query12 = "Letter: What is \"refractory\" cardiac failure?"
	query13 = "Step-up therapy for children with uncontrolled asthma receiving inhaled corticosteroids."
	query14 = "angel dust. PCP."
	query15= "(vascular endothelial growth factors)"
	text1 = """
		brain magnetic resonance imaging (MRI) scans
	"""
	query16 = "page"
	query17="feeling cold"
	query18="DPA - docosapentaenoic acid"
	query19="randomized clinical trials"
	query20="aid"
	query21="computed tomography"
	query22="examination of something"
	query23="examination of the kawasaki"
	query24="vehicle"
	query25="cells"
	query26="IL-11."
	query27="bare metal stent"
	query28="percutaneous transluminal pulmonary"
	query29="above knee amputation AKA"
	query30="uncontrolled asthma"
	query31="V-ABC protein"
	query32="Cough and chest pain. RCT."
	query33="interleukin (IL)-6"
	query34="Sustained recovery of progressive multifocal leukoencephalopathy after treatment with IL-2."
	query35="interleukin-1 receptor antagonist"
	query36="achr"
	query37="Pulmonary veins. PVs."
	query38="Atrial fibrillation is the most common sustained cardiac arrhythmia, and contributes greatly to cardiovascular morbidity and mortality and congestive heart failure"
	query39="findings from the Baltimore ECA."
	query40 = "bare metal stent"
	query41 = "ECG"
	query42="lung adenocarcinoma"
	query43="ovary cancer"
	query44="Everolimus an inhibitor of the"
	query45="Aortic dissection"
	query46="Likelihood ratio"
	query47="E. coli"
	query48="cold"
	query49="hepatitis B"
	query50="Reduced plasma concentrations of nitrogen oxide in individuals with essential hypertension"
	query51="protein kinase C"
	query52="Renal replacement therapy"
	query53="Methotrexate can improve joint pain in rheumatoid arthritis"
	query54="coronavirus disease 2019"
	query55="glycoside hydrolase (GH) family"
	query56="Vitamin C sepsis"
	query57="hungry bone syndrome"
	query58="Prospective observational cohort study"
	query59="Inhaled nitric oxide NO"
	query60="intraoperative floppy iris syndrome"
	query61="We conclude that the loss of vagal tone associated with the development of cardiac failure unmasks the direct negative chronotropic effect of exogenous adenosine on the sinoatrial node"
	query62="Combination of tocolytic agents for inhibiting preterm labour"
	query63="Things are seldom what they seem"
	query64="luteinizing hormone releasing hormone"
	query65="T cell"
	query66="IVDU"
	query67="bioprosthetic mitral valve"
	query68="Asthma exacerbation India"
	query69="From the Centers for Disease Control and Prevention"
	query70="advanced primary cutaneous cutaneous T cell"
	query71="cows walked on the prairie"
	query72="Remdesivir for the Treatment of Covid-19 - Final Report."
	query73="CAR T cell"
	query74="antipsychotic"
	query75="Efficacy and Safety of Once-Daily Insulin Degludec/Insulin Aspart versus Insulin Glargine (U100) for 52 Weeks in Insulin-Naïve Patients with Type 2 Diabetes"
	query76="Over 12 years, 49 patients with hyperparathyroidism secondary to chronic renal failure under treatment with hemodialysis were treated with total parathyroidectomy"
	query77="With nephrocalcinosis and without nephrocalcinosis (P < .01)"
	query78="Hidradenitis suppurativa is a painful chronic inflammatory skin disease with few options for effective treatment"
	query79="Indication for immunotherapy includes evidence of IgE mediated disease"
	query80="Randomized controlled trials of patients with acute myocardial infarction"
	query81="right and left"
	query82="hepatic and splenic blush on computed tomography in children following blunt force acute interstitial nephritis"
	query83="White matter lesions (WMLs) are commonly found on brain MRI of migraine patients"
	query84="Helicobacter pylori"
	query85="The percutaneous coronary intervention PCI have to be one of the fastest growing therapeutic interventions for patients with ST elevation myocardial infarction"
	query86="kaposi's sarcoma"
	query87="COPD Insulin"
	query88="distal DVT"
	query89="luteinizing hormone releasing hormone deficiency in thyroid hormone deficiency"
	query90 = "Antibiotic concentrations used for exposure were either the MIC of each agent for the sensitive isolates or the recommended sensitivity breakpoint concentrations for the resistant isolates"
	query91="biomarker"
	unittest.main()
	
	# counter = 0
	# d = u.Timer('t')

	# lmtzr = WordNetLemmatizer()

	# while (counter < 1):
	# 	term = query91
	# 	term = clean_text(term)

	# 	all_words = get_all_words_list(term, lmtzr)

	# 	spellcheck_threshold = 100

	# 	cache = get_cache(all_words_list=all_words, case_sensitive=True, \
	# 		check_pos=False, spellcheck_threshold=spellcheck_threshold, lmtzr=lmtzr)

	# 	sentences_df = pd.DataFrame([[term, 'title', 0,0]], \
	# 		columns=['line', 'section', 'section_ind', 'ln_num'])
	# 	item = pd.DataFrame([[term, 'title', 0, 0]], columns=['line', 'section', 'section_ind', 'ln_num'])

	# 	res, g, s = annotate_text_not_parallel(sentences_df=sentences_df, cache=cache, \
	# 		case_sensitive=True, check_pos=False, bool_acr_check=False,\
	# 		spellcheck_threshold=spellcheck_threshold, \
	# 		write_sentences=True, lmtzr=None)

	# 	d.stop()
	# 	u.pprint(g['sentence_tuples'].tolist())
	# 	u.pprint(res)

	# 	counter += 1
	
	
