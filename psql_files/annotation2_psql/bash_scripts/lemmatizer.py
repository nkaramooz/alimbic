import sys
sys.path.append('../../../utilities')
sys.path.append('../../../')
import pandas as pd
import pglib as pg
import sqlalchemy as sqla
import snomed_annotator2 as ann
import utils2 as u
import sys
from nltk.stem.wordnet import WordNetLemmatizer


def lemma(word):
	lmtzr = WordNetLemmatizer()
	return lmtzr.lemmatize(word)


# should change this to use the clean_text function in snomed_annotator
def lemmatize(row):
	terms = row['term']
	terms = terms.replace(' - ', ' ').replace('.', '').replace('- ', ' ').replace(' -', ' ').replace('-', ' ').replace(',', '').replace('\'\'', ' ').replace('   ', ' ').replace('  ', ' ').rstrip().lstrip()

	ln_words = terms.split()
	lemmas = ann.get_lemmas(ln_words, True)

	try:
		row['word'] = lemmas[row['word_ord']-1]
		return row
	except:
		u.pprint(row)
		print(lemmas)
		print(terms)
		sys.exit(0)

def lemmatize_table():
	query = """
		select 
			adid
			,acid
			,term
			,lower(term) as term_lower
			,word
			,word_ord
			,term_length
			,is_acronym 
		from annotation2.add_adid_acronym
		"""
	conn, cursor = pg.return_postgres_cursor()

	new_candidate_df = pg.return_df_from_query(cursor, query, None, \
		['adid', 'acid', 'term', 'term_lower', 'word', 'word_ord', 'term_length', 'is_acronym'])

	new_candidate_df.loc[~new_candidate_df.word.isin(['vs', 'as']), 'word'] = new_candidate_df.loc[~new_candidate_df.word.isin(['vs', 'as'])].apply(lemmatize, axis=1)

	engine = pg.return_sql_alchemy_engine()

	new_candidate_df.to_sql('lemmas', \
		engine, schema='annotation2', if_exists='replace', index=False)

	index_query = """
		set schema 'annotation2';
		create index lemmas_cid_ind on lemmas(acid);
		create index lemmas_did_ind on lemmas(adid);
		create index lemmas_term_ind on lemmas(term);
		create index lemmas_word_ind on lemmas(word);
		create index lemmas_word_ord_ind on lemmas(word_ord);
		create index lemmas_term_lower_ind on lemmas(term_lower);
	"""

	cursor.execute(index_query, None)
	cursor.connection.commit()
	cursor.close()


if __name__ == "__main__":
	lemmatize_table()