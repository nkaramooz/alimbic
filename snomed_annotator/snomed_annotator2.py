# TODO: annotate_line is still pretty inefficient and should be refactored.
import pandas as pd
import json
from fuzzywuzzy import fuzz
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk.data
import numpy as np
from utilities import pglib as pg
import uuid
import sqlite3

# Given a word, the function will return a list of candidate concepts.
# This function will only grab concepts where the first word of the concept
# matches the word provided, which is a limitation of this implementation.
# If fuzzy_match is true, it will leverage soundex and levenshtein to find
# candidate concepts.
def get_new_candidate_df(word, case_sensitive, fuzzy_match):
	new_candidate_query = ""
	new_candidate_df = None

	if fuzzy_match:
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
				where soundex(query_word) = tb2.soundex
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
				where soundex(query_word) = tb2.soundex
			"""
		new_candidate_df = pg.return_df_from_query(new_candidate_query, (list(word),), \
	 		["a_did", "a_cid", "term", "word", "word_ord", "term_length", "is_acronym"])

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
		new_candidate_df = pg.return_df_from_query(new_candidate_query, (tuple(word),), \
		 ["a_did", "a_cid", "term", "word", "word_ord", "term_length", "is_acronym"])

	return new_candidate_df


def get_reverse_lemmas():
	with open("./utilities/custom_lemmas.json", 'r') as lem:
		custom_lemmas = json.load(lem)
		reverse_lemmas = {}
		for k,v in custom_lemmas.items():
			if k not in reverse_lemmas.keys():
				reverse_lemmas[v] = [k]
			else:
				reverse_lemmas[v].append(k)
		return reverse_lemmas
	

def get_lemmas(ln_words, case_sensitive, check_pos, lmtzr):
	pos_tag = get_wordnet_pos(ln_words) if check_pos else None
	lmtzr = WordNetLemmatizer() if lmtzr is None else lmtzr
	ln_lemmas = []
	with open("./utilities/custom_lemmas.json", 'r') as lem:
		custom_lemmas = json.load(lem)

		for i,w in enumerate(ln_words):
			# If the case_sensitive=True, we will avoid lowercasing words that are acronyms
			w = w.lower() if not case_sensitive else w.lower() if w.upper() != w else w

			skip_words = ["als","vs", "assess", "as", "whiting", "pylori", "fasting", "ankylosing"]
			
			if w.upper() == w or w in skip_words:
				pass
			elif w in custom_lemmas.keys():
				w = custom_lemmas[w]
			else:
				w = lmtzr.lemmatize(w, pos_tag[i]) if check_pos else lmtzr.lemmatize(w)
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


# Line should be cleaned before being passed to this function.
# Returns words_df and annotations_df
# words_df is a dataframe with the following columns: term_index, term, line
# annotations_df is a dataframe that has all the concepts that were found in the line
# annotations_df may be None if no concepts were found for the entire line
def snomed_annotation(line, spellcheck_threshold, case_sensitive, check_pos, cache, lmtzr):
	annotation_header = ['query', 'substring', 'substring_start_index', 'substring_end_index', 'a_cid', 'is_acronym']
	ln_words = line.split()
	ln_lemmas = get_lemmas(ln_words, case_sensitive, check_pos, lmtzr)

	# No candidates for this line, so only words_df will have values
	if len(cache) == 0:
		words_df = pd.DataFrame()

		for index,word in enumerate(ln_lemmas):
			words_df = pd.concat([words_df, pd.DataFrame([[index, word]], columns=['term_index', 'term'])])
		words_df['line'] = line
		return words_df, None

	candidate_df_arr = []
	results_df = pd.DataFrame()
	words_df = pd.DataFrame()
	annotations_df = pd.DataFrame()

	# Iterate through words in the line.
	# If there are active matches as represented by entries in candidate_df_arr,
	# then evaluate the new word against the active matches.
	for index, word in enumerate(ln_lemmas):
		words_df = pd.concat([words_df, pd.DataFrame([[index, word]], columns=['term_index', 'term'])], ignore_index=True)

		if len(candidate_df_arr) > 0:
			candidate_df_arr = evaluate_candidate_df(word, index, candidate_df_arr, spellcheck_threshold, case_sensitive)
		
		new_adid_list = get_candidate_adids(cache, word, spellcheck_threshold)
		
		if len(new_adid_list) > 0:
			new_candidate_df = cache[cache.a_did.isin(new_adid_list)].copy()
			if spellcheck_threshold != 100:
				# For fuzzy matching, words appear as columns in the cache.
				# Set order_distance to 100 for words that achieve spellcheck threshold.
				# Only use word_ord=1 in case the first word of a concept is repeated in the concepts full description.
				new_candidate_df.loc[(new_candidate_df[word] >= spellcheck_threshold) & (new_candidate_df['word_ord']==1), 'order_dist'] = 100.0
			else:
				new_candidate_df.loc[new_candidate_df['word'] == word, 'order_dist'] = 100.0
			
			new_candidate_df.loc[new_candidate_df['order_dist'] != 100.0, 'order_dist'] = -1.0
			new_candidate_df['description_start_index'] = index
			
			candidate_df_arr.append(new_candidate_df)
		
		# Extract descriptions where all the words in the description found a match (new_results_df)
		candidate_df_arr, new_results_df = get_results(candidate_df_arr, index)
		results_df = pd.concat([results_df, new_results_df], ignore_index=True)
	
	if len(results_df.index) > 0:
		order_score = results_df
		order_score = order_score[['a_cid', 'a_did', 'description_start_index', 'description_end_index', 'term', 'order_dist']].groupby(\
			['a_cid', 'a_did', 'description_start_index'], as_index=False)['order_dist'].sum()
		order_score.rename(columns={'order_dist' : 'final_score'}, inplace=True)
		results_df = results_df.join(order_score.set_index(['a_cid', 'a_did', 'description_start_index']), on=['a_cid', 'a_did', 'description_start_index'])
		annotations_df = prune_results(results_df, results_df)
	
	words_df['line'] = line
	
	return words_df, annotations_df


# Helper function for line_annotation
# The limitation here is that it won't catch instances where first and second word
# of a phrase at the beginning of the line are reversed.
def get_candidate_adids(cache, word, spellcheck_threshold):
	if spellcheck_threshold != 100:
		match_df = cache[cache.word_ord == 1].copy()
		match_df['fuzz'] = match_df['word'].apply(lambda x: fuzz.ratio(word, x))
		match_df = match_df[match_df['fuzz'] >= spellcheck_threshold]
	else:
		match_df = cache[(cache['word'] == word) & (cache.word_ord == 1)]
	
	return match_df['a_did'].tolist()


# candidate_df_arr is an array of dataframes that contain active matches.
# This function will return a filtered array of dataframes that have an ongoing match
# If a description does not have any values of "-1.0" in the order_dist column, then
# the description is considered complete.
def get_results(candidate_df_arr, end_index):
	new_candidate_df_arr = []
	results_df = pd.DataFrame()

	for index, df in enumerate(candidate_df_arr):
		exclusion_series = df[df['order_dist'] == -1.0]['a_did'].tolist()
		new_results = df[~df['a_did'].isin(exclusion_series)].copy()
		new_results['description_end_index'] = end_index
		
		remaining_candidates = df[df['a_did'].isin(exclusion_series)]
		if len(remaining_candidates.index) > 0:
			new_candidate_df_arr.append(remaining_candidates)
		results_df = pd.concat([results_df, new_results], ignore_index=True)
	return new_candidate_df_arr,results_df


# This function takes the annotations from an entire document and uses it to
# arbitrate how acronyms should be interpreted. 
# TODO: Add exceptions here for common acronyms that don't need support in the document.
def acronym_check(results_df):
	acronym_df = results_df[results_df['is_acronym'] == True].copy()
	if len(acronym_df.index) > 0:
		non_acronyms_df = results_df[results_df['is_acronym'] == False].copy()
		
		# Grouping so that we don't count a concept id multiple times just
		# because it has multiple tokens that add up to the concept.
		non_acronyms_df = non_acronyms_df[['a_cid', 'description_start_index', 'section_ind', 'ln_num']].drop_duplicates()
		cid_cnt_df = non_acronyms_df['a_cid'].value_counts().reset_index()
		
		acronym_df = acronym_df.merge(cid_cnt_df, on=['a_cid'],how='left')
		approved_acronyms = acronym_df[acronym_df['count'] >= 1]
		
		final = pd.concat([results_df[results_df['is_acronym'] == False].copy(), approved_acronyms], ignore_index=True)
		return final
	
	return results_df


# candidate_df_arr has the array of dataframes that contain the candidate concepts
def evaluate_candidate_df(word, substring_start_index, candidate_df_arr, spellcheck_threshold, case_sensitive):
	new_candidate_df_arr = []
	for index, df in enumerate(candidate_df_arr):
		df_copy = df.copy()

		if spellcheck_threshold != 100:
			# Want to penalize concepts where the word deviates from the word sequence
			# in the concept description. Continue to assume that the first word in a concept
			# appears first in the text.

			# First figure out the offset for the current match. a_did, offset
			offset = df_copy[df_copy['word_ord']==1][['a_did', 'description_start_index']]
			offset.rename(columns={'description_start_index' : 'offset'}, inplace=True)

			offset['offset'] = offset['offset'].apply(lambda x: x-1 if x > 1 else x)
			offset['offset'] = offset['offset'].apply(lambda x: 0.0 if x == 1 else x)
			offset['offset'] = offset['offset'].apply(lambda x: 1.0 if x == 0 else x)
			
			new_order_dist = df_copy[(df_copy[word] >= spellcheck_threshold) & (df_copy['order_dist'] == -1.0)].copy()
			new_order_dist = new_order_dist.join(offset.set_index('a_did'), on=['a_did'])
			new_order_dist['tmp_new_order_dist'] = (substring_start_index + new_order_dist['offset'] - new_order_dist['word_ord'])
			
			new_order_dist['new_order_dist'] = new_order_dist[['tmp_new_order_dist', 'order_dist']].max(1)
			new_order_dist['new_order_dist'] = new_order_dist['new_order_dist'].apply(lambda x: 100 if x == 0 else x)
			new_order_dist = new_order_dist[['a_did', 'word', 'word_ord','new_order_dist']]
			
			df_copy = df_copy.join(new_order_dist.set_index(['a_did', 'word', 'word_ord']), on=['a_did', 'word', 'word_ord'])
			df_copy['order_dist'] = df_copy.new_order_dist.combine_first(df_copy.order_dist)
			df_copy = df_copy.drop(columns=['new_order_dist'])
			new_candidate_df_arr.append(df_copy)
		else:
			word_ord = df_copy[(df_copy['word'] == word) & (df_copy['order_dist'] == -1.0)].groupby(['a_did'], as_index=False)['word_ord'].min()
			word_ord['tmp'] = 1.0
			df_copy = df_copy.merge(word_ord, on=['a_did', 'word_ord'], how='left')
			df_copy.loc[(df_copy.order_dist == -1.0) & (df_copy['word'] == word)
				& (df_copy['tmp'] == 1.0), \
				['substring_start_index']] = substring_start_index
			df_copy.loc[(df_copy.order_dist == -1.0) & (df_copy['word'] == word) & (df_copy['tmp'] == 1.0), \
				['order_dist']] = 100
			df_copy = df_copy.drop(columns=['tmp'])
			new_candidate_df_arr.append(df_copy)

	# Now want to make sure that number has gone up from before
	final_candidate_df_arr = []

	for index, new_df in enumerate(new_candidate_df_arr):
		new_df_description_score = new_df.groupby(['a_did'], as_index=False)['order_dist'].sum()
		old_df_description_score = candidate_df_arr[index].groupby(['a_did'], as_index=False)['order_dist'].sum()

		if len(old_df_description_score.index) > 0 and len(new_df_description_score.index) > 0:
			candidate_descriptions = new_df_description_score[new_df_description_score['order_dist'] > old_df_description_score['order_dist']]
			filtered_candidates = new_df[new_df['a_did'].isin(candidate_descriptions['a_did'])]
			if len(filtered_candidates.index) != 0:
				final_candidate_df_arr.append(filtered_candidates)

	return final_candidate_df_arr


# This function takes the list of concepts that were annotated
# and attempts to adjudicate to a single concept per term.
# The purpose here is that the phrase "Chronic obstructive pulmonary disease"
# is annotated with a single concept, and does not have the concept "Chronic"
# annotated to the phrase.
# TODO: Move this into a while loop instead of a recursive function
def prune_results(scores_df, og_results):
	# exclude_index_arr houses the data frame indices of items
	# that have already been reviewed and should not be evaluated again.
	exclude_index_arr = []
	scores_df = scores_df.sort_values(['description_start_index'], ascending=True)
	
	results_df = pd.DataFrame()
	results_index = []
	changes_made = False

	# Convert to dict to speed up iterating through annotations
	scores_dict = scores_df.to_dict('index')
	
	# index of the dictionary is the same as the index for the dataframe
	for index in scores_dict:
		row = scores_dict[index]

		# If the index is already excluded, then continue to the next index
		if index not in exclude_index_arr:
			exclude = scores_df.index.isin(exclude_index_arr)
			
			# Remove excluded items
			subset_df = scores_df[~exclude].sort_values(['final_score', 'term_length'])
			
			# Find all the concepts that overlap with the indices of the current concept
			# subset_df will include data from th current row in the dictionary
			subset_df = subset_df[
  				((subset_df['description_start_index'] <= row['description_start_index']) 
  					& (subset_df['description_end_index'] >= row['description_end_index'])) 
  				| ((subset_df['description_start_index'] <= row['description_end_index']) 
  					& (subset_df['description_end_index'] >= row['description_end_index']))
  				| ((subset_df['description_start_index'] >= row['description_start_index'])
  					& ((subset_df['description_end_index'] <= row['description_end_index'])))]
			
			subset_df = subset_df.sort_values(['final_score', 'term_length'], ascending=False)

			# Use the result with the highest final score.
			result = subset_df.head(1).copy()
			
			# Exclude all items in subset for future evaluation
			if len(subset_df.index) > 1:
				changes_made = True
				new_exclude = subset_df
				exclude_index_arr.append(index)
				exclude_index_arr.extend(new_exclude.index.values)

			# Add the result to the ongoing results dataframe
			results_df = pd.concat([results_df, result], ignore_index=True)

	# Return the results if no new items were evaluated
	if not changes_made:
		final_results = og_results.merge(results_df[['description_start_index','description_end_index']], how='inner', \
								   left_on=['description_start_index','description_end_index'], \
									right_on=['description_start_index','description_end_index'])
		final_results = final_results[['a_did', 'a_cid', 'term', 'word', 'word_ord', 'term_length', 'is_acronym', 'description_start_index', 'description_end_index', 'final_score']]
		return final_results
	else:
		return prune_results(results_df, og_results)


# Resolves instances where both two concepts selected because
# Their score is completely the same.
def resolve_conflicts(results_df):

	acronym_df = results_df[results_df['is_acronym'] == True].copy()
	results_df = results_df[results_df['is_acronym'] == False].copy()

	counts_df = results_df.drop_duplicates(['description_start_index','a_cid', 'section_ind', 'ln_num']).copy()
	counts_df['index_count'] = 1
	counts_df = counts_df.groupby(['description_start_index', 'section_ind', 'ln_num'], as_index=False)['index_count'].sum()
	
	# description_start_index with more than one concept annotation
	conflicted_indices = counts_df[counts_df['index_count'] > 1]

	conflict_free_df = results_df[~((results_df['description_start_index'].isin(conflicted_indices['description_start_index'])) & \
		(results_df['ln_num'].isin(conflicted_indices['ln_num'])) &\
		(results_df['section_ind'].isin(conflicted_indices['section_ind'])))].copy()
	conflict_free_df['concept_count'] = 1
	# Collect the problematic rows in one data frame.
	conflicted_df = results_df[(results_df['description_start_index'].isin(conflicted_indices['description_start_index'])) & \
		(results_df['ln_num'].isin(conflicted_indices['ln_num'])) & \
		(results_df['section_ind'].isin(conflicted_indices['section_ind']))].copy()
	final_results = conflict_free_df.copy()
	
	# Attempt to favor interpretations that have additional support
	# in the document.

	
	concept_weights = conflict_free_df.groupby(['a_cid'], as_index=False)['concept_count'].sum()
	join_weights = conflicted_df.merge(concept_weights, on=['a_cid'],how='left')

	#set below to negative 10 and drop all rows with negative value
	join_weights.fillna({'concept_count' : 0}, inplace=True)
	join_weights['final_score'] = join_weights['final_score'] + join_weights['concept_count']
		
	# Reason to sort by a_cid is to have something slightly deterministic in the event both scores are the same
	join_weights = join_weights.sort_values(['section_ind', 'ln_num', 'description_start_index', 'final_score', 'a_cid'], ascending=False)
		
	# Track which description_start_index was last evaluated
	last_term_start_index = None
	last_ln_num = None
	last_section_ind = None
	
	#TODO: Consider if the concept_counts should be updated with each new interpretation
	for index, row in join_weights.iterrows():
		if last_term_start_index != row['description_start_index'] or last_ln_num != row['ln_num'] or \
			last_section_ind != row['section_ind']:
			final_results = pd.concat([final_results, row.to_frame().T], ignore_index=True)
			last_term_start_index = row['description_start_index']
			last_ln_num = row['ln_num']
			last_section_ind = row['section_ind']
	final_results = pd.concat([final_results, acronym_df], ignore_index=True)

	final_results.drop(columns=['concept_count'], inplace=True)
	return final_results


def get_preferred_concept_names(a_cid, cursor):
	search_query = "select acid, term from annotation2.preferred_concept_names \
			where acid = %s"
	params = (a_cid,)
	names_df = pg.return_df_from_query(cursor, search_query, params, ['a_cid', 'term'])
	return names_df['term'][0]


def add_names(results_df, cursor):
	if results_df is None:
		return None
	else:
		search_query = "select acid, term from annotation2.preferred_concept_names \
			where acid in %s"
		params = (tuple(results_df['a_cid']),)
		names_df = pg.return_df_from_query(cursor, search_query, params, ['a_cid', 'term'])
		results_df = results_df.merge(names_df, on='a_cid')

		return results_df


# For errors check to make sure that get_reverse_lemmas has reverse lemmas
def get_original_line_tuples(line, cleaned_tuples, case_sensitive, lmtzr):
	line_words = line.split()
	max_index = len(line_words)
	line_words_lemmas = get_lemmas(line_words, case_sensitive, True, lmtzr)
	current_line_index = 0
	og_annotated_tuples = []
	prev_og_word = ""
	reverse_lemmas = get_reverse_lemmas()
	reverse_lemmas_keys = reverse_lemmas.keys()
	split_counter = 0
	prev_annotated_word = ""
	# If item[0] is the same as previous item[0] the algorithm is not certain
	# on the annotation, so both should be included.
	for index, item in enumerate(cleaned_tuples):

		if current_line_index < max_index:

			if prev_annotated_word == item[0]:
				og_annotated_tuples.append((line_words[current_line_index-1], item[1]))
			else:

				cleaned_word = item[0].lower()
				og_word = line_words_lemmas[current_line_index].lower()
				og_set = set(clean_text(og_word).split())
				last_word_set = set(clean_text(prev_og_word).split())
				max_split = len(last_word_set)-1

				if cleaned_word in reverse_lemmas_keys:
					cleaned_words_plus_lemmas = [cleaned_word]
					cleaned_words_plus_lemmas.extend(reverse_lemmas[cleaned_word])
					cleaned_words_plus_lemmas = set(cleaned_words_plus_lemmas)
				else:
					new_forms = [cleaned_word]
					for lem in wordnet.lemmas(cleaned_word):
						for new_word in lem.derivationally_related_forms():
							new_forms.append(new_word.name())
					cleaned_words_plus_lemmas = set(new_forms)

				# Return arbitrary string that wont appear in text. Very inelegant.
				if len(last_word_set.intersection(cleaned_words_plus_lemmas)) > 0 and split_counter < max_split:
					split_counter += 1
					prev_og_word = line_words_lemmas[current_line_index-1].lower()
				elif len(og_set.intersection(cleaned_words_plus_lemmas)) > 0:
					og_annotated_tuples.append((line_words[current_line_index], item[1]))
					current_line_index+=1
					prev_og_word = og_word
					split_counter = 0
				else:
					og_annotated_tuples.append((line_words[current_line_index], 0))
					current_line_index += 1
					last_word = og_word
					split_counter = 0
				prev_annotated_word = item[0]

	return og_annotated_tuples


# sentence dict fields: line, section, section_ind (index of section), ln_num
def annotate_line(sentence_dict, case_sensitive, check_pos, cache, \
		spellcheck_threshold, lmtzr):	
	words_df, annotation_df = snomed_annotation(line=sentence_dict['line'], \
								spellcheck_threshold=spellcheck_threshold,\
								case_sensitive=case_sensitive, check_pos=check_pos, cache=cache, lmtzr=lmtzr)

	if annotation_df is not None:
		annotation_df['ln_num'] = sentence_dict['ln_num']
		annotation_df['section'] = sentence_dict['section']
		annotation_df['section_ind'] = sentence_dict['section_ind']
	
	words_df['ln_num'] = sentence_dict['ln_num'] if annotation_df is not None else 0
	words_df['section'] = sentence_dict['section']
	words_df['section_ind'] = sentence_dict['section_ind']

	return words_df, annotation_df


# Returns the sentence tuple for the annotation.
# The first element of the tuple is the term, and the second element is the concept id.
# If there is no annotated concept id for the term, the second element is 0.
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
			if isinstance(row['a_cid'], str):
				sentence_arr.append((row['term'], row['a_cid']))
			else:
				sentence_arr.append((row['term'], 0))
		return sentence_arr
	
	else:
		return None


# When child_candidates is provided, it does not expand the query
# but rather reformats arrays around primary cids and filters
# to only child a_cids that exist in the search results
def query_expansion(conceptid_series, flattened_concept_list, child_candidates, cursor):

	conceptid_tup = tuple(flattened_concept_list)
	if child_candidates is None and len(conceptid_tup) > 0:

		# may need to artificially limit query size at some point for 
		# elastic search
		child_query = """
			select
				t1.child_acid
				,t1.parent_acid
			from snomed2.transitive_closure_acid t1
			where child_acid in (select acid from annotation2.used_descriptions) and parent_acid in %s
			limit 1000
		"""
		child_df = pg.return_df_from_query(cursor, child_query, (conceptid_tup,), \
			["child_a_cid", "parent_a_cid"])

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
			["child_a_cid", "parent_a_cid"])

	else:
		return []

	results_list = []
	for a_cid_list in conceptid_series:
		sub_results_list = []
		for item in a_cid_list:
			temp_res = [item]
			added_other = False
			# JUST CHANGED PARENTHESES location

			if len(child_df[child_df['parent_a_cid'] == item].index) > 0:
				temp_res.extend(child_df[child_df['parent_a_cid'] == item]['child_a_cid'].tolist())
				added_other = True

			if added_other:
				sub_results_list.append(temp_res)
			else:
				sub_results_list.append(temp_res)
		if len(sub_results_list) > 0:
			results_list.append(sub_results_list)

	return results_list


## removed limit 15 for generating training dataset
def get_children(conceptid, cursor):
	child_query = """
		select 
			child_acid
		from snomed2.transitive_closure_acid tb1
		left join annotation2.concept_counts tb2
			on tb1.child_acid = tb2.concept
		where tb1.parent_acid = %s and tb2.cnt is not null
		order by tb2.cnt desc
	"""
	child_df = pg.return_df_from_query(cursor, child_query, (conceptid,), \
		["child__cid"])
	

	return child_df['child_a_cid'].tolist()


def get_all_words_list(text):
	tokenized = nltk.sent_tokenize(text)
	all_words = list(set([w for line in tokenized for w in line.split() if w.strip() != '']))
	return all_words


# check_pos is a boolean to flag if wordnet's part of speech tagger should be used
# to disambiguate words with multiple meanings.
# spellcheck_threshold is a value from 0-100 that determines the threshold for
# fuzzy matching.
# Returns empty data frame if no matches are found.
# For fuzzy matching, the cache data frame will produce a dataframe with a column for
# each word in the list with values showing the similarity to the word in the concept.
def get_cache(all_words_list, case_sensitive, check_pos, spellcheck_threshold, lmtzr):
	lemmas = list(set(get_lemmas(all_words_list, case_sensitive, check_pos, lmtzr)))

	# cache obtains the candidate concepts for the distinct words provided, which
	# speeds up the annotation process.
	cache = get_new_candidate_df(lemmas, case_sensitive, spellcheck_threshold != 100)

	if len(cache.index) > 0:
		if spellcheck_threshold != 100:
			for word in lemmas:
				cache[word] = cache['word'].apply(lambda x: fuzz.ratio(word, x))
			cache['max_fuzz'] = cache[lemmas].max(axis=1)

			# For all words that are above the threshold, we want to give them a point.
			cache.loc[cache['max_fuzz'] >= spellcheck_threshold, 'points'] = 1.0
			cache.loc[cache['max_fuzz'] < spellcheck_threshold, 'points'] = 0.0
		else:
			cache['points'] = 0.0
			cache.loc[cache.word.isin(lemmas), 'points'] = 1.0

		weights = cache[['a_did', 'term_length', 'points']].groupby(['a_did', 'term_length'], as_index=False)['points'].sum()
		# Filter to concepts where the term length is the same as the number of points.
		candidate_dids = weights[weights['term_length'] == weights['points']]['a_did'].tolist()
		cache = cache[cache['a_did'].isin(candidate_dids)].copy()
		# Ignore errors since max_fuzz only exists for fuzzy matching
		cache = cache.drop(['points', 'max_fuzz'], axis=1, errors='ignore')
		
		return cache
	
	return pd.DataFrame()


# Returns 3 dataframes to be eventually materialized into tables
# sentence_annotations_df colums=['id (sentnece)', 'section', 'section_ind', 'ln_num', 'a_cid', 'a_did', 'final_ann']
# ann_df returned for annotation if write sentences is negative
# otherwise returns 3 dataframes for pubmed indexing
# TODO: Clean this up to avoid neeeding sqlite3.
# acr_check evaluates for acronyms and looks for supporting material in the text to favor
# one interpretation over another.
# check_pos boolean to flag if wordnet's part of speech tagger should be used.
def annotate_text(sentences_df, cache, case_sensitive, \
	check_pos, acr_check, write_sentences, lmtzr, spellcheck_threshold):
	# concepts_df contains one row per concept in the provided sentence with 
	# the start and stop indices.
	concepts_df = pd.DataFrame()

	sentence_df = pd.DataFrame()

	# words_df contains one row per word in the provided sentence with the 
	# corresponding concept annotation at the word level.
	words_df = pd.DataFrame()

	#TODO avoid needing to convert to dictionary
	sentence_dict = sentences_df.to_dict('records')

	for row in sentence_dict:
		line_words_df, res_df = annotate_line(sentence_dict=row, case_sensitive=case_sensitive, \
			check_pos=check_pos, cache=cache, spellcheck_threshold=spellcheck_threshold, lmtzr=lmtzr)
		
		if res_df is not None:
			concepts_df = pd.concat([concepts_df, res_df], ignore_index=True)
		words_df = pd.concat([words_df, line_words_df], ignore_index=True)

	if len(concepts_df.index) > 0:
		concepts_df = resolve_conflicts(concepts_df)
		
		if acr_check:
			concepts_df = acronym_check(concepts_df)

	concepts_df = concepts_df.drop_duplicates(subset=['a_cid', 'description_start_index', 'section_ind', 'ln_num']).copy()
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
				,t2.a_cid
				,t2.a_did
				,t2.description_end_index
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
		ann_df['a_cid'] = np.nan
		ann_df['a_did'] = np.nan
		ann_df.insert(0, 'description_start_index', 0, 0+len(ann_df))
		ann_df.insert(0, 'description_end_index', 0, 0+len(ann_df))

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
						concept_arr = list(set(ln_df['a_cid'].dropna().tolist()))
						uid = str(uuid.uuid1())
					
						annotated_tuples = get_annotated_tuple(ln_df)

						ln_df['sentence_id'] = uid
						ln_df.fillna(-1, inplace=True)
						
						ln_df.to_sql('ln_df', conn, index=False, if_exists='replace')
						query = """
							select 
								sentence_id
								,section
								,section_ind
								,ln_num
								,a_cid
								,a_did
								,case when a_cid='-1' then term else a_cid end as final_ann
								,line
							from ln_df
							group by sentence_id,section,section_ind,ln_num, description_start_index, description_end_index, final_ann
						"""
						sentence_annotations_df = pd.concat([sentence_annotations_df, pd.read_sql_query(query,conn)], ignore_index=True)
						single_ln_df = ln_df[['sentence_id', 'section', 'section_ind', 'ln_num', 'line']].copy()
						
						try: 
							single_ln_df = single_ln_df.head(1).copy()
						except:
							print("ERROR")
							print(single_ln_df)
							print(ann_df)
						single_ln_df['sentence_tuples'] = [annotated_tuples]
						single_ln_df['og_sentence_tuples'] = [get_original_line_tuples(single_ln_df['line'].values[0], annotated_tuples, case_sensitive, lmtzr)]
						sentence_tuples_df = pd.concat([sentence_tuples_df, single_ln_df], ignore_index=True)
						single_concept_arr_df = single_ln_df[['sentence_id', 'section', 'section_ind', 'ln_num']].copy()
						single_concept_arr_df['concept_arr'] = [concept_arr]
						sentence_concept_arr_df = pd.concat([sentence_concept_arr_df, single_concept_arr_df], ignore_index=True)
		return sentence_annotations_df, sentence_tuples_df, sentence_concept_arr_df

	return ann_df


def clean_text(line):
	# Order here does matter
	chars_to_remove = ['.(','.', '!',',',';','*','[',']','-',':','"',':','(',')','\\','/','  ']
	for c in chars_to_remove:
		line = line.replace(c, ' ')
	return line