import http.client
import pandas as pd
import sys
from bs4 import BeautifulSoup, SoupStrainer
import requests
import string
import snomed_annotator2 as ann2
from search.views import return_section_sentences
import utilities.pglib as pg
import sqlalchemy as sqla
import wikipediaapi

def get_wiki_concept_links(categorymembers, wiki_wiki, counter):

	for c in categorymembers.values():
		print(c.title, c.ns)
		if c.ns == 14 and 'deaths from' not in c.title.lower() and 'alcohol' not in c.title.lower()\
			and 'cannabis' not in c.title.lower() and 'doping' not in c.title.lower() and counter < 3:
			print(c.title, 'category')
			new_cat = wiki_wiki.page(c.title)
			print(new_cat.categorymembers)
			get_wiki_concept_links(new_cat.categorymembers, wiki_wiki, counter+1)
		elif c.ns == 0 and 'deaths from' not in c.title.lower() \
			and 'hospital' not in c.title.lower() and 'president' not in c.title.lower() \
			and 'cannabis' not in c.title.lower() and 'doping' not in c.title.lower() :
			try:
				print(c)
				page_url = c.fullurl
				print(page_url)
				check_entity_url(page_url)
			except:
				continue

def check_entity_url(entity_url):
	response = requests.get(url=entity_url)
	soup = BeautifulSoup(response.content, 'html.parser')
	for header in soup.find_all('h1'):
		header_text = header.text
		header_text = ann2.clean_text(header_text)
		header_words = ann2.get_all_words_list(header_text)

		cache = ann2.get_cache(all_words_list=header_words, case_sensitive=True, \
			check_pos=False, spellcheck_threshold=100, lmtzr=None)

		header_df = return_section_sentences(header_text, 'header', 0, pd.DataFrame())

		header_concepts_df = ann2.annotate_text_not_parallel(sentences_df=header_df, cache=cache, \
			case_sensitive=True, check_pos=False, bool_acr_check=False,\
			spellcheck_threshold=100, \
			write_sentences=False, lmtzr=None)


		if len(set(header_concepts_df['acid'].tolist())) == 1 and \
			len(header_concepts_df[header_concepts_df['acid'].isna()]) == 0:

			a_cid = header_concepts_df['acid'].tolist()[0]

			paragraphs = soup.find_all('p')
			if len(paragraphs) > 1:
				paragraph = paragraphs[1].text

				description = paragraph.split('.')[0]
				query = """
					select term
					from annotation2.preferred_concept_names
					where acid=%s
				"""
				conn,cursor = pg.return_postgres_cursor()
				preferred_name = pg.return_df_from_query(cursor, query, (a_cid,), ['term'])
				if len(preferred_name) == 0:
					query = """
						select term
						from annotation2.downstream_root_did
						where acid=%s order by length(term) desc limit 1
					"""
					preferred_name = pg.return_df_from_query(cursor, query, (a_cid,), ['term'])['term'][0]
				else:
					preferred_name = preferred_name['term'][0]
				conn.close()
				cursor.close()
				entity_df = pd.DataFrame([[a_cid, preferred_name, description]], 
					columns=['a_cid', 'term', 'desc'])

				engine = pg.return_sql_alchemy_engine()
				entity_df.to_sql('wiki_spacy_entities', engine, schema='spacy', if_exists='append', index=False)
				engine.dispose()


if __name__ == "__main__":
	# get_wiki_concept_links()
	# check_entity_url("https://en.wikipedia.org/wiki/Myocardial_infarction")
	wiki_wiki = wikipediaapi.Wikipedia('en')
	cat = wiki_wiki.page("Category:Drugs")

	get_wiki_concept_links(cat.categorymembers, wiki_wiki, 0)
	