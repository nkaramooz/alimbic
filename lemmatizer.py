from nltk.stem.wordnet import WordNetLemmatizer
import utils as u, pglib as pg


def lemma(word):
	lmtzr = WordNetLemmatizer()
	return lmtzr.lemmatize(word)

def lemmatize_description_id(description_id, cursor):
	query = """
		select * from annotation.augmented_active_selected_concept_key_words_v2
		where description_id = '%s'
	""" % description_id

	new_candidate_df = pg.return_df_from_query(cursor, query, None, \
		['description_id', 'conceptid', 'term', 'word', 'word_ord', 'term_length'])
	new_candidate_df['word'] = new_candidate_df['word'].map(lemma)

	engine = pg.return_sql_alchemy_engine()
	new_candidate_df.to_sql('augmented_active_selected_concept_key_words_lemmas_2', \
		engine, schema='annotation', if_exists='append', index=False)



if __name__ == '__main__':

	# query = "select * from annotation.augmented_active_selected_concept_key_words_v2"
	cursor = pg.return_postgres_cursor()

	# new_candidate_df = pg.return_df_from_query(cursor, query, None, \
	# 	['description_id', 'conceptid', 'term', 'word', 'word_ord', 'term_length'])



	# new_candidate_df['word'] = new_candidate_df['word'].map(lemma)

	# engine = pg.return_sql_alchemy_engine()

	# new_candidate_df.to_sql('augmented_active_selected_concept_key_words_lemmas_2', \
	# 	engine, schema='annotation', if_exists='replace', index=False)

	index_query = """
		set schema 'annotation';
		create index lemmas_conceptid_ind on augmented_active_selected_concept_key_words_lemmas_2(conceptid);
		create index lemmas_description_id_ind on augmented_active_selected_concept_key_words_lemmas_2(description_id);
		create index lemmas_term_ind on augmented_active_selected_concept_key_words_lemmas_2(term);
		create index lemmas_word_ind on augmented_active_selected_concept_key_words_lemmas_2(word);
		create index lemmas_word_ord_ind on augmented_active_selected_concept_key_words_lemmas_2(word_ord);
	"""

	cursor.execute(index_query, None)
	cursor.connection.commit()