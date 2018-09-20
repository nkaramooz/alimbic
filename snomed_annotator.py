import pandas as pd
import re
import pickle
import sys
import psycopg2
from sqlalchemy import create_engine
from fuzzywuzzy import fuzz
from nltk.stem.wordnet import WordNetLemmatizer
import nltk.data
import numpy as np
import time
import multiprocessing as mp
import copy
import utilities.utils as u, utilities.pglib as pg
import unittest
import uuid

def get_new_candidate_df(word, cursor, case_sensitive):

	#Knocks off a second off a sentence by selecting for word_ord=1
	new_candidate_query = ""

	if case_sensitive:
		new_candidate_query = "select description_id, conceptid, term, word, word_ord, term_length, \
			case when word = %s then 1 else 0 end as l_dist from annotation.lemmas_3 where \
			description_id in \
			(select description_id from annotation.lemmas_3 where word = %s and \
				(word_ord = 1))"
	else:
		new_candidate_query = "select description_id, conceptid, term, word, word_ord, term_length, \
			case when word = %s then 1 else 0 end as l_dist from annotation.lemmas_3 where \
			description_id in \
			(select description_id from annotation.lemmas_3 where word = %s and \
				(word_ord = 1))"

	new_candidate_df = pg.return_df_from_query(cursor, new_candidate_query, (word, word), \
	 ["description_id", "conceptid", "term", "word", "word_ord", "term_length", "l_dist"])
	
	return new_candidate_df

def return_line_snomed_annotation_v2(cursor, line, threshold, case_sensitive, cache):
	
	annotation_header = ['query', 'substring', 'substring_start_index', 'substring_end_index', 'conceptid']	
	annotation_header = ['query', 'substring', 'substring_start_index', \
		'substring_end_index', 'conceptid']

	line = line
	ln_words = line.split()

	candidate_df_arr = []
	results_df = pd.DataFrame()

	lmtzr = WordNetLemmatizer()

	for index,word in enumerate(ln_words):

		# identify acronyms
		if word.upper() != word:
			word = word.lower()

		if word.lower() != 'vs':
			word = lmtzr.lemmatize(word)
		candidate_df_arr, active_match = evaluate_candidate_df(word, index, candidate_df_arr, threshold, case_sensitive)
		
		# if not active_match:
		new_candidate_df = pd.DataFrame()
		if word in cache.keys():
			new_candidate_df = cache[word]
		else:
			new_candidate_df = get_new_candidate_df(word, cursor, case_sensitive)
			cache[word] = new_candidate_df.copy()
		if len(new_candidate_df.index) > 0:
			new_candidate_df['substring_start_index'] = index
			new_candidate_df['description_start_index'] = index
			candidate_df_arr.append(new_candidate_df)
					
		candidate_df_arr, new_results_df = get_results(candidate_df_arr, index)
		results_df = results_df.append(new_results_df, sort=False)
	# u.pprint(results_df)
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

		# u.pprint(joined_results)
		final_results = prune_results_v2(joined_results, joined_results)
		
		if len(final_results.index) > 0:
			final_results['line'] = line

			return final_results, cache

	return None, cache

def get_results(candidate_df_arr, end_index):
	## no description here
	new_candidate_df_arr = []
	results_df = pd.DataFrame()
	for index,df in enumerate(candidate_df_arr):
		exclusion_series = df[df['l_dist'] == 0]['description_id'].tolist()
		new_results = df[~df['description_id'].isin(exclusion_series)].copy()
		new_results['description_end_index'] = end_index
		
		remaining_candidates = df[df['description_id'].isin(exclusion_series)]
		if len(remaining_candidates.index) != 0:
			new_candidate_df_arr.append(remaining_candidates)
		results_df = results_df.append(new_results, sort=False)

	return new_candidate_df_arr,results_df

# May need to add exceptions for approved acronyms / common acronyms that don't need support
def acronym_check(results_df):
	results_df['acronym'] = np.where(results_df['term'] == results_df['term'].str.upper(), True, False)
	non_acronyms_df = results_df[results_df['acronym'] == False].copy()
	
	cid_counts = non_acronyms_df['conceptid'].value_counts()
	cid_cnt_df = pd.DataFrame({'conceptid':cid_counts.index, 'count':cid_counts.values})

	acronym_df = results_df[results_df['acronym'] == True].copy()
	acronym_df = acronym_df.merge(cid_cnt_df, on=['conceptid'],how='left')
	approved_acronyms = acronym_df[acronym_df['count'] >= 1]
	final = non_acronyms_df.append(approved_acronyms, sort=False)
	return final
	



def evaluate_candidate_df(word, substring_start_index, candidate_df_arr, threshold, case_sensitive):

	new_candidate_df_arr = []
	for index,df in enumerate(candidate_df_arr):

		df_copy = df.copy()
		for index,row in df_copy.iterrows():

			l_dist = float()
			if case_sensitive:
				l_dist = fuzz.ratio(word, row['word'])
			else:
				l_dist = fuzz.ratio(word.lower(), row['word'].lower())

			
			# assign l_dist to only those that pass threshold
			if l_dist >= threshold: ### TUNE ME
				df_copy.loc[index, 'l_dist'] = l_dist/100.00
				df_copy.loc[index, 'substring_start_index'] = substring_start_index


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
		final_results = pd.DataFrame()
		for index,row in results_df.iterrows():

			expanded_res = og_results[ \
				(og_results['term_start_index'] == row['term_start_index']) \
				& (og_results['term_end_index'] == row['term_end_index'])]
			
			final_results = final_results.append(expanded_res, sort=False)
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
		conc_count_query = "select conceptid, cnt from annotation.concept_counts where conceptid in %s"
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
	annotation, cache = return_line_snomed_annotation_v2(cursor, line, 93, case_sensitive, cache)

	if annotation is not None:
		annotation['ln_number'] = ln_number
		
	return annotation, cache

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
			subtypeid as conceptid
			,supertypeid
		from (
			select subtypeid, supertypeid
			from snomed.curr_transitive_closure_f where supertypeid in %s
		) tb
		join annotation.concept_counts ct
		  on tb.subtypeid = ct.conceptid
		order by ct.cnt desc
		limit 1
	"""

	child_df = pg.return_df_from_query(cursor, child_query, (conceptid_tup,), \
		["conceptid", "supertypeid"])

	results_list = []

	for item in conceptid_series:
		temp_res = [item]
		added_other = False
		# JUST CHANGED PARENTHESES location
		if len(syn_df[syn_df['reference_conceptid'] == item].index) > 0:
			temp_res.extend(syn_df[syn_df['reference_conceptid'] == item]['reference_conceptid'].tolist())
			added_other = True

		if len(child_df[child_df['supertypeid'] == item].index) > 0:
			temp_res.extend(child_df[child_df['supertypeid'] == item]['conceptid'].tolist())
			added_other = True

		if added_other:
			results_list.append(temp_res)
		else:
			results_list.extend(temp_res)

	return results_list


def annotate_text_not_parallel(text, section, cursor, case_sensitive):

	tokenized = nltk.sent_tokenize(text)
	ann_df = pd.DataFrame()
	sentence_df = pd.DataFrame(columns=['id', 'conceptid', 'concept_arr', 'section', 'line_num', 'sentence', 'sentence_tuples'])
	
	cache = {}

	for ln_number, line in enumerate(tokenized):
		res_df, cache = annotate_line_v2(line, ln_number, cursor, case_sensitive, cache)
		
		ann_df = ann_df.append(res_df, sort=False)

	if len(ann_df.index) > 0:
		ann_df = resolve_conflicts(ann_df, cursor)
		ann_df = acronym_check(ann_df)

		for ln_number, line in enumerate(tokenized):
			ln_df =  ann_df[ann_df['ln_number'] == ln_number].copy()
			concept_arr = list(set(ln_df['conceptid'].tolist()))
			u = str(uuid.uuid1())
			s_arr = get_sentence_annotation(line, ln_df)

			for cid in concept_arr:
				sentence_df = sentence_df.append(pd.DataFrame([[u, cid, concept_arr, section, ln_number, line, s_arr]], 
						columns=['id', 'conceptid', 'concept_arr', 'section', 'line_num', 'sentence', 'sentence_tuples']), sort=False)
		return ann_df, sentence_df
	else:
		return None, None


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
	# line = line.replace('(', '')
	# line = line.replace(')', '')
	line = re.sub("\(.*?\)","",line)
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
	query38="Atrial fibrillation is the most common sustained cardiac arrhythmia, and contributes greatly to cardiovascular morbidity and mortality. Many aspects of the management of atrial fibrillation remain controversial. We address nine specific controversies in atrial fibrillation management, briefly focusing on the relations between mechanisms and therapy, the roles of rhythm and rate control, the definition of optimum rate control, the need for early cardioversion to prevent remodelling, the comparison of electrical with pharmacological cardioversion, the selection of patients for long-term oral anticoagulation, the roles of novel long-term anticoagulation approaches and ablation therapy, and the potential usefulness of upstream therapy targeting substrate development. The background of every controversy is reviewed and our opinions expressed. Here, we hope to inform physicians about the most important controversies in this specialty and stimulate investigators to address unresolved issues."
	query39="findings from the Baltimore ECA."
	query40 = "bare metal stent"
	query41 = "ECG"
	query42="lung adenocarcinoma"
	query43="ovary cancer"
	query44="Everolimus an inhibitor of the "
	check_timer = u.Timer("full")

	# pprint(add_names(return_query_snomed_annotation_v3(query, 87)))
	conn, cursor = pg.return_postgres_cursor()

	# u.pprint(return_line_snomed_annotation(cursor, query1, 87))
	# u.pprint(return_line_snomed_annotation(cursor, query2, 87))
	# u.pprint(return_line_snomed_annotation(cursor, query3, 87))
	# query 11
	res, sentences = annotate_text_not_parallel(query43, 'title', cursor, False)
	cursor.close()
	u.pprint("=============================")
	u.pprint(res)
	u.pprint(sentences)
	check_timer.stop()

	# u.pprint("*****************************")
	# unittest.main()
	
