import sys
sys.path.append('../../../utilities')
sys.path.append('../../..')
import pandas as pd
import pglib as pg
import sqlalchemy as sqla
import snomed_annotator2 as ann2
import utils2 as u
from nltk.stem.wordnet import WordNetLemmatizer

lmtzr = WordNetLemmatizer()

def lemma(ln):

	term = ln['term']
	term = term.replace(' - ', ' ').replace('.', '').replace('- ', ' ').replace(' -', ' ').replace('-', ' ').replace(',', '').replace('\'\'', ' ').replace('   ', ' ').replace('  ', ' ').rstrip().lstrip()
	word_ord = ln['word_ord']
	ln_words = term.split()
	ln_lemmas = ann2.get_lemmas(ln_words, True, True, lmtzr)
	try:
		return ln_lemmas[word_ord-1]
	except:
		print(term)
		print(ln_words)
		print(ln_lemmas)
		u.pprint(ln)
		sys.exit(0)

def update_lemmas():
	
	query = """
		select 
			adid
			,acid
			,term
			,term_lower
			,word
			,word_ord
			,term_length
			,is_acronym 
		from annotation2.add_adid_acronym
		where acid in
			(select distinct(root_acid) from annotation2.concept_types 
			where rel_type in ('condition', 'study_design', 'chemical', 'symptom', 'prevention', 'treatment', 'statistic', 'diagnostic', 'outcome', 'cause', 'anatomy') 
				and active != 0)
		"""
	conn, cursor = pg.return_postgres_cursor()

	new_candidate_df = pg.return_df_from_query(cursor, query, None, \
		['adid', 'acid', 'term', 'term_lower', 'word', 'word_ord', 'term_length', 'is_acronym'])

	new_candidate_df.loc[~new_candidate_df.word.isin(['vs', 'as']), 'word'] = new_candidate_df.loc[~new_candidate_df.word.isin(['vs', 'as'])].apply(lemma, axis=1)
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
	update_lemmas()