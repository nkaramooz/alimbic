import pandas as pd
import re
import sys
import psycopg2
from fuzzywuzzy import fuzz
from nltk.stem.wordnet import WordNetLemmatizer
import nltk.data
import numpy as np
# import multiprocessing as mp
import copy
import utilities.utils as u
import utilities.pglib as pg
import unittest
import uuid
import time
import sys

def get_new_candidate_df(word, case_sensitive, cursor):

	#Knocks off a second off a sentence by selecting for word_ord=1
	new_candidate_query = ""

	if case_sensitive:
		new_candidate_query = """
			select 
				description_id
				,conceptid
				,term
				,tb1.word
				,word_ord
				,term_length
				,is_acronym
			from annotation.lemmas_3 tb1
			join annotation.first_word tb2
			  on tb1.description_id = ANY(tb2.did_agg)
			where tb2.word in %s
		"""
	else:
		for i,v in enumerate(word):
			word[i] = word[i].lower()

		new_candidate_query = """
			select 
				description_id
				,conceptid
				,term
				,lower(tb1.word) as word
				,word_ord
				,term_length
				,is_acronym
			from annotation.lemmas_3 tb1
			join annotation.first_word tb2
			  on tb1.description_id = ANY(tb2.did_agg)
			where lower(tb2.word) in %s
		"""


	new_candidate_df = pg.return_df_from_query(cursor, new_candidate_query, (tuple(word),), \
	 ["description_id", "conceptid", "term", "word", "word_ord", "term_length", "is_acronym"])

	# new_candidate_df[(new_candidate_df.word_ord==2) & (new_candidate_df.word=='syndrome')]

	return new_candidate_df

def return_line_snomed_annotation_v2(cursor, line, threshold, case_sensitive, cache):

	annotation_header = ['query', 'substring', 'substring_start_index', 'substring_end_index', 'conceptid', 'is_acronym']

	ln_words = line.split()

	candidate_df_arr = []
	results_df = pd.DataFrame()

	lmtzr = WordNetLemmatizer()

	# c = u.Timer('get_candidates')
	c = u.Timer("iterate_candidates")

	for index,word in enumerate(ln_words):
		# identify acronyms

		if not case_sensitive:
			word = word.lower()
		else:
			if word.upper() != word:
				word = word.lower()

		if word.lower() != 'vs':
			word = lmtzr.lemmatize(word)

		j = u.Timer("evaluate_candidate")
		candidate_df_arr, active_match = evaluate_candidate_df(word, index, candidate_df_arr, threshold, case_sensitive)

		# if not active_match:

		new_candidate_df = cache[cache.word == word]

		if len(new_candidate_df.index) > 0:
			new_candidate_ids = new_candidate_df[new_candidate_df.word_ord == 1]['description_id']
			# print(cache[(cache.word == word) & (cache.word_ord == 1)])
			new_candidate_df = cache[cache.description_id.isin(new_candidate_ids)].copy()
			new_candidate_df['l_dist'] = -1.0
			new_candidate_df.loc[new_candidate_df.word == word, 'l_dist'] = 1

		# else:
		# 	new_candidate_df = get_new_candidate_df(word, cursor, case_sensitive)
		# 	cache[word] = new_candidate_df.copy()
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

		order_score = order_score[['conceptid', 'description_id', 'description_start_index', 'description_end_index', 'term', 'order_score']].groupby(\
			['conceptid', 'description_id', 'description_start_index'], as_index=False)['order_score'].sum()
	
		distinct_results = results_df[['conceptid', 'description_id', 'description_start_index', 'description_end_index', 'term']].drop_duplicates()
	
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
		joined_results['final_score'] = joined_results['final_score'] + 0.5*joined_results['term_length']



		final_results = prune_results_v2(joined_results, joined_results)
		final_results['is_acronym'] = final_results.merge(results_df, on=['description_id'])['is_acronym']

		if len(final_results.index) > 0:
			final_results['line'] = line

			return final_results

	return None

def get_results(candidate_df_arr, end_index):
	## no description here
	new_candidate_df_arr = []
	results_df = pd.DataFrame()
	for index,df in enumerate(candidate_df_arr):
		exclusion_series = df[df['l_dist'] == -1.0]['description_id'].tolist()
		new_results = df[~df['description_id'].isin(exclusion_series)].copy()
		new_results['description_end_index'] = end_index
		
		remaining_candidates = df[df['description_id'].isin(exclusion_series)]
		if len(remaining_candidates.index) > 0:
			new_candidate_df_arr.append(remaining_candidates)
		results_df = results_df.append(new_results, sort=False)
	
	return new_candidate_df_arr,results_df

# May need to add exceptions for approved acronyms / common acronyms that don't need support
def acronym_check(results_df):
	non_acronyms_df = results_df[results_df['is_acronym'] == 0].copy()
	
	cid_counts = non_acronyms_df['conceptid'].value_counts()
	cid_cnt_df = pd.DataFrame({'conceptid':cid_counts.index, 'count':cid_counts.values})

	acronym_df = results_df[results_df['is_acronym'] == 1].copy()
	acronym_df = acronym_df.merge(cid_cnt_df, on=['conceptid'],how='left')
	approved_acronyms = acronym_df[acronym_df['count'] >= 1]
	final = non_acronyms_df.append(approved_acronyms, sort=False)
	return final

# def l_func(row, word):
	

def evaluate_candidate_df(word, substring_start_index, candidate_df_arr, threshold, case_sensitive):
	threshold = threshold/100.0

	new_candidate_df_arr = []
	for index,df in enumerate(candidate_df_arr):

		df_copy = df.copy()
		df_copy['l_dist_tmp'] = -1.0
		if case_sensitive:
			df_copy['l_dist_tmp'] = df_copy['word'].apply(lambda x: fuzz.ratio(x, word)/100.0)

		else:
			df_copy['l_dist_tmp'] = df_copy['word'].apply(lambda x: fuzz.ratio(x.lower(), word.lower())/100.0)

		df_copy.loc[(df_copy.l_dist == -1.0) & (df_copy.l_dist_tmp >= threshold), ['substring_start_index']] = substring_start_index
		df_copy.loc[(df_copy.l_dist == -1.0) & (df_copy.l_dist_tmp >= threshold), ['l_dist']] = df_copy[(df_copy.l_dist == -1.0) & (df_copy.l_dist_tmp >= threshold)][['l_dist_tmp']].values
		df_copy = df_copy.drop(['l_dist_tmp'], axis=1)
	
		new_candidate_df_arr.append(df_copy)


	# now want to make sure that number has gone up from before
	# ideally should also at this point be pulling out complete matches.

	final_candidate_df_arr = []
	active_match = False

	for index, new_df in enumerate(new_candidate_df_arr):
		new_df_description_score = new_df.groupby(['description_id'], as_index=False)['l_dist'].sum()
		old_df_description_score = candidate_df_arr[index].groupby(['description_id'], as_index=False)['l_dist'].sum()



		if len(old_df_description_score.index) > 0 and len(new_df_description_score.index) > 0:
			candidate_descriptions = new_df_description_score[new_df_description_score['l_dist'] > old_df_description_score['l_dist']]
			filtered_candidates = new_df[new_df['description_id'].isin(candidate_descriptions['description_id'])]
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

# Attempts to disambiguate terms to select one (acronym disambiguation)
# Technically currently in the first IF statement if there is still a tie, it should resort
# to using previous concept frequencies. Examples where this would pertain is where a 
# document references pneumocystis pneumonia and phencyclidine, both of which have the
# acronym PCP
# will defer fixing this till later
def resolve_conflicts(results_df, cursor):
	results_df['index_count'] = 1
	counts_df = results_df.groupby(['term_start_index', 'ln_number'], as_index=False)['index_count'].sum()
	
	conflicted_indices = counts_df[counts_df['index_count'] > 1]
	conflict_free_df = results_df[~((results_df['term_start_index'].isin(conflicted_indices['term_start_index'])) & \
		(results_df['ln_number'].isin(conflicted_indices['ln_number'])))].copy()
	conflicted_df = results_df[(results_df['term_start_index'].isin(conflicted_indices['term_start_index'])) & \
		(results_df['ln_number'].isin(conflicted_indices['ln_number']))].copy()
	
	final_results = conflict_free_df.copy()

	if len(conflict_free_df.index) > 0:
		
		conflict_free_df['concept_count'] = 1
		concept_weights = conflict_free_df.groupby(['conceptid'], as_index=False)['concept_count'].sum()

		join_weights = conflicted_df.merge(concept_weights, on=['conceptid'],how='left')

		#set below to negative 10 and drop all rows with negative value
		join_weights['concept_count'].fillna(0, inplace=True)
		join_weights['final_score'] = join_weights['final_score'] + join_weights['concept_count']
		join_weights = join_weights.sort_values(['ln_number', 'term_start_index', 'final_score'], ascending=False)

		last_term_start_index = None
		last_ln_number = None

		for index,row in join_weights.iterrows():
			if last_term_start_index != row['term_start_index'] or last_ln_number != row['ln_number']:
				final_results = final_results.append(row, sort=False)
				last_term_start_index = row['term_start_index']
				last_ln_number = row['ln_number']

	# Really should only want below for query annotation. Not document annotation
	else: #first try and choose most common concept. If not choose randomly
		conc_count_query = "select conceptid, count from annotation.concept_counts where conceptid in %s"
		params = (tuple(conflicted_df['conceptid']),)
		cid_cnt_df = pg.return_df_from_query(cursor, conc_count_query, params, ['conceptid', 'cnt'])


		conflicted_df = conflicted_df.merge(cid_cnt_df, on=['conceptid'], how='left')
		conflicted_df['cnt'] = conflicted_df['cnt'].fillna(value=1)

		conflicted_df['final_score'] = conflicted_df['final_score'] * conflicted_df['cnt']

		conflicted_df = conflicted_df.sort_values(['ln_number', 'term_start_index', 'final_score'], ascending=False)

		last_term_start_index = None
		last_ln_number = None

		for index,row in conflicted_df.iterrows():
			if last_term_start_index != row['term_start_index'] or last_ln_number != row['ln_number']:
				final_results = final_results.append(row, sort=False)
				last_term_start_index = row['term_start_index']
				last_ln_number = row['ln_number']
	return final_results


def add_names(results_df):
	conn,cursor = pg.return_postgres_cursor()
	if results_df is None:
		cursor.close()
		return None
	else:
		# Using old table since augmented tables include the acronyms
		search_query = "select conceptid, term from annotation.preferred_concept_names \
			where conceptid in %s"

		params = (tuple(results_df['conceptid']),)
		names_df = pg.return_df_from_query(cursor, search_query, params, ['conceptid', 'term'])

		results_df = results_df.merge(names_df, on='conceptid')
		cursor.close()
		conn.close
		return results_df


def annotate_line_v2(line, ln_number, cursor, case_sensitive, cache):

	
	line = clean_text(line)
	annotation = return_line_snomed_annotation_v2(cursor, line, 93, case_sensitive, cache)
	if annotation is not None:
		annotation['ln_number'] = ln_number
		
	return annotation

def get_sentence_annotation(line, c_df):
	c_df = c_df.sort_values(by=['description_start_index'], ascending=True)
	if len(c_df) >= 1:
		c_res = pd.DataFrame()
		sentence_arr = []

		line = clean_text(line)
		ln_words = line.split()
		concept_counter = 0
		concept_len = len(c_df)
		at_end = False
		for ind1, word in enumerate(ln_words):
			added = False
			while (concept_counter < concept_len):
				if ((ind1 >= c_df.iloc[concept_counter]['description_start_index']) and (ind1 <= c_df.iloc[concept_counter]['description_end_index'])):
					sentence_arr.append((word, c_df.iloc[concept_counter]['conceptid']))
					added = True
					if ((ind1 == c_df.iloc[concept_counter]['description_end_index'])):
						concept_counter +=1
					break
				else:
					sentence_arr.append((word, 0))
					added = True
					break

			if not added:
				sentence_arr.append((word, 0))

		return sentence_arr
	else:
		return None

def get_concept_synonyms_list_from_list(concept_list, cursor):
	query = "select synonym_conceptid from annotation.concept_terms_synonyms where reference_conceptid in %s"
	synonym_df = pg.return_df_from_query(cursor, query, (tuple(concept_list),), ["synonym_conceptid"])
	concept_list.extend(synonym_df['synonym_conceptid'].tolist())
	return list(set(concept_list))

def get_concept_synonyms_list_from_series(conceptid_series, cursor):

	conceptid_list = tuple(conceptid_series.tolist())
	
	query = "select reference_conceptid, synonym_conceptid from annotation.concept_terms_synonyms where reference_conceptid in %s"

	synonym_df = pg.return_df_from_query(cursor, query, (conceptid_list,), \
	 ["reference_conceptid", "synonym_conceptid"])

	results_list = []

	for item in conceptid_series:
		if len(synonym_df[synonym_df['reference_conceptid'] == item].index) > 0:
			sub_list = [item]
			sub_df = synonym_df[synonym_df['reference_conceptid'] == item]
			for ind,val in sub_df.iterrows():
				sub_list.append(val['synonym_conceptid'])
			results_list.append(sub_list)
		else:
			results_list.append(item)

	return results_list

def get_concept_synonyms_df_from_series(conceptid_series, cursor):
	conceptid_list = tuple(conceptid_series.tolist())
	query = "select reference_conceptid, synonym_conceptid from annotation.concept_terms_synonyms where reference_conceptid in %s"

	synonym_df = pg.return_df_from_query(cursor, query, (conceptid_list,), \
	 ["reference_conceptid", "synonym_conceptid"])

	return synonym_df

def query_expansion(conceptid_series, cursor):
	conceptid_tup = tuple(conceptid_series.tolist())
	syn_query = "select reference_conceptid, synonym_conceptid from annotation.concept_terms_synonyms where reference_conceptid in %s"
	syn_df = pg.return_df_from_query(cursor, syn_query, (conceptid_tup,), \
	 ["reference_conceptid", "synonym_conceptid"])

	child_query = """
		select 
			conceptid
			,supertypeid
		from (
			select
				supertypeid,
    			conceptid,
    			count,
    			row_number() over (partition by supertypeid order by count desc) as rn
			from
			(
				select 
					supertypeid
            		,subtypeid as conceptid
            		,ct.count
				from (
					select subtypeid, supertypeid
					from snomed.curr_transitive_closure_f where supertypeid in %s
				) tb1
				join annotation.concept_counts ct
		  			on tb1.subtypeid = ct.conceptid
    		) tb2
		) tb3
		where rn <= 4
	"""

	child_df = pg.return_df_from_query(cursor, child_query, (conceptid_tup,), \
		["conceptid", "supertypeid"])

	results_list = []

	for item in conceptid_series:
		temp_res = [item]
		added_other = False
		# JUST CHANGED PARENTHESES location

		if len(syn_df[syn_df['reference_conceptid'] == item].index) > 0:
			temp_res.extend(syn_df[syn_df['reference_conceptid'] == item]['synonym_conceptid'].tolist())
			added_other = True

		if len(child_df[child_df['supertypeid'] == item].index) > 0:
			temp_res.extend(child_df[child_df['supertypeid'] == item]['conceptid'].tolist())
			added_other = True

		if added_other:
			results_list.append(temp_res)
		else:
			results_list.extend(temp_res)

	return results_list

def get_children(conceptid, cursor):
	child_query = """
		select 
			subtypeid as conceptid
		from snomed.curr_transitive_closure_f tb1
		left join annotation.concept_counts tb2
		on tb1.subtypeid = tb2.conceptid
		where supertypeid = %s and tb2.count is not null
		order by tb2.count desc
		limit 15
	"""
	child_df = pg.return_df_from_query(cursor, child_query, (conceptid,), \
		["conceptid"])
	

	return child_df['conceptid'].tolist()

def get_all_words_list(text):
	tokenized = nltk.sent_tokenize(text)
	all_words = []
	lmtzr = WordNetLemmatizer()
	for ln_number, line in enumerate(tokenized):
		words = line.split()
		for index,w in enumerate(words):
			if w.upper() != w:
				w = w.lower()

			if w.lower() != 'vs':
				w = lmtzr.lemmatize(w)
			all_words.append(w)

	return list(set(all_words))

def get_cache(all_words_list, case_sensitive, cursor):
	cache = get_new_candidate_df(all_words_list, case_sensitive, cursor)
	cache['points'] = 0

	cache.loc[cache.word.isin(all_words_list), 'points'] = 1
	csf = cache[['description_id', 'term_length', 'points']].groupby(['description_id', 'term_length'], as_index=False)['points'].sum()

	candidate_dids = csf[csf['term_length'] == csf['points']]['description_id'].tolist()
	
	cache = cache[cache['description_id'].isin(candidate_dids)].copy()
	cache = cache.drop(['points'], axis=1)
	return cache

def annotate_text_not_parallel(text, section, cache, cursor, case_sensitive, bool_acr_check, write_sentences):

	tokenized = nltk.sent_tokenize(text)
	ann_df = pd.DataFrame()
	sentence_df = pd.DataFrame(columns=['id', 'conceptid', 'concept_arr', 'section', 'line_num', 'sentence', 'sentence_tuples'])

	for ln_number, line in enumerate(tokenized):
		res_df = annotate_line_v2(line, ln_number, cursor, case_sensitive, cache)
		ann_df = ann_df.append(res_df, sort=False)



	if len(ann_df.index) > 0:
		# No significant time sink below
		ann_df = resolve_conflicts(ann_df, cursor)

		if write_sentences:
			for ln_number, line in enumerate(tokenized):
				ln_df =  ann_df[ann_df['ln_number'] == ln_number].copy()
				concept_arr = list(set(ln_df['conceptid'].tolist()))
				u = str(uuid.uuid1())
				s_arr = get_sentence_annotation(line, ln_df)

				for cid in concept_arr:
					sentence_df = sentence_df.append(pd.DataFrame([[u, cid, concept_arr, section, ln_number, line, s_arr]], 
							columns=['id', 'conceptid', 'concept_arr', 'section', 'line_num', 'sentence', 'sentence_tuples']), sort=False)
	return ann_df, sentence_df



### UTILITY FUNCTIONS

def clean_text(line):
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

	# This makes you lose the likelihood ratios
	# line = re.sub("\(.*?\)","",line)
	return line

def timeLimit(t, ref):
	return (t <= ref)

class TestAnnotator(unittest.TestCase):

	def test_equal(self):
		queryArr = [("protein c deficiency protein s deficiency", 10, ['1233987014', '1221214016']),
			("chronic obstructive pulmonary disease and congestive heart failure", 22, ['475431013', '70653017'])]


		cursor = pg.return_postgres_cursor()
		for q in queryArr:
			check_timer = u.Timer(q[0])
		

			res = annotate_text_not_parallel(q[0], 'title', cursor, False)
			u.pprint("=============================")
			
			
			u.pprint(res)
			t = check_timer.stop_num()
			d = ((q[1]-t)/t)*100
			self.assertTrue(timeLimit(t, q[1]))
			print(res['description_id'].values)
			self.assertTrue(set(res['description_id'].values) == set(q[2]))
			print(t)
			u.pprint("============================")

	

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
		Cough at night asthma congestion sputum
	"""
	query4 = """
		H2-receptor antagonists
	"""

	query5 = """
		Heart Failure with Preserved Ejection Fraction.
	"""

	query6 = """
		Treatment
	"""

	query7 = """
		Weekly vs. Every-3-Week Paclitaxel and Carboplatin for Ovarian Cancer
	"""

	query8 = """
		Diastolic heart failure--abnormalities in active relaxation and passive stiffness of the left ventricle.
	"""
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
	query53="Methotrexate can improve joint pain in rheumatoid arthritis" ## NEED TO FIX THIS. NOT ANNOTATING CORRECTLY
	query54="coronavirus disease 2019"
	query55="glycoside hydrolase (GH) family"
	query56="Vitamin C sepsis"
	conn, cursor = pg.return_postgres_cursor()


	counter = 0
	while (counter < 1):
		d = u.Timer('t')
		term = query56
		term = clean_text(term)
		all_words = get_all_words_list(term)
		cache = get_cache(all_words, False, cursor)
		res, sentences = annotate_text_not_parallel(term, 'title', cache, cursor, False, True, False)
		# u.pprint(res)
		res = acronym_check(res)
		u.pprint(res)
		
		d.stop()
		counter += 1
	
	# cursor.close()
	# u.pprint("=============================")
	# u.pprint(res)
	# u.pprint(sentences)
	# check_timer.stop()
	# u.pprint(get_children('387458008', cursor))
	# labeled_set = [['418285008','387458008', '1']]
	# labelled_set = pd.DataFrame()
	# for index,item in enumerate(labeled_set):
		
	# 	root_cids = [item[0]]
	# 	root_cids.extend(get_children(item[0], cursor))
	# 	print(root_cids)

	# u.pprint("*****************************")
	# unittest.main()
	
