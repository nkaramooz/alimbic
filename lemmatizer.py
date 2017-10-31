from nltk.stem.wordnet import WordNetLemmatizer
import utils as u, pglib as pg


def lemma(word):
	lmtzr = WordNetLemmatizer()
	return lmtzr.lemmatize(word)


if __name__ == '__main__':

	query = "select * from annotation.augmented_active_selected_concept_key_words"
	cursor = pg.return_postgres_cursor()
	c = u.Timer('get_df')
	new_candidate_df = pg.return_df_from_query(cursor, query, None, \
		['description_id', 'conceptid', 'term', 'word', 'word_ord', 'term_length'])
	c.stop()

	d = u.Timer('apply_func')
	new_candidate_df['word'] = new_candidate_df['word'].map(lemma)

	engine = pg.return_sql_alchemy_engine()

	new_candidate_df.to_sql('augmented_active_selected_concept_key_words_lemmas', engine, schema='annotation', if_exists='replace')
