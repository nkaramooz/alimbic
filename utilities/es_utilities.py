import snomed_annotator as ann
import utilities.pglib as pg
from elasticsearch import Elasticsearch
from nltk.stem.wordnet import WordNetLemmatizer
import utilities.utils as u
import pandas as pd

# Create your views here.

INDEX_NAME='pubmed3'

def update_postgres_document_concept_count():
	scroller = ElasticScroll({'host' : 'localhost', 'port' : 9200})
	
	conceptid_df = pd.DataFrame()
	# while (scroller.has_next):
	

	while scroller.has_next:
		article_list = scroller.next()

		for hit in article_list:
			abstract_arr = get_flattened_abstract_concepts(hit)
			if abstract_arr is not None:
				conceptid_df = conceptid_df.append(pd.DataFrame(abstract_arr, columns=['conceptid']), \
					ignore_index=True)

			if hit['_source']['title_conceptids'] is not None:

				conceptid_df = conceptid_df.append(pd.DataFrame(hit['_source']['title_conceptids'], \
					columns=['conceptid']), ignore_index=True)

	conceptid_df['count'] = 1
	conceptid_df = conceptid_df.groupby(['conceptid'], as_index=False)['count'].sum()

	engine = pg.return_sql_alchemy_engine()
	conceptid_df.to_sql('concept_counts', engine, schema='annotation', if_exists='replace')


class ElasticScroll():
	def __init__(self,server_info):
		self.es = Elasticsearch([server_info])
		self.initialized = False
		self.sid = None
		self.scroll_size = None
		self.has_next = True

	def next(self):
		if not self.initialized:
			pages = self.es.search(index='pubmed3', doc_type='abstract', scroll='1000m', \
				size=1000, body={"query" : {"match_all" : {}}})
			self.sid = pages['_scroll_id']
			self.scroll_size = pages['hits']['total']
			self.initialized = True
			return pages['hits']['hits']
		else:
			if self.scroll_size > 0:
				pages = self.es.scroll(scroll_id = self.sid, scroll='1000m')
				self.sid = pages['_scroll_id']
				self.scroll_size = len(pages['hits']['hits'])
				if self.scroll_size == 0:
					self.has_next = False
				return pages['hits']['hits']


def get_flattened_abstract_concepts(hit):
	abstract_concept_arr = []
	try:
		for key1 in hit['_source']['abstract_conceptids']:
			for key2 in hit['_source']['abstract_conceptids'][key1]:
				concept_df = pd.DataFrame(hit['_source']['abstract_conceptids'][key1][key2], columns=['conceptid'])
				concept_df = ann.add_names(concept_df)
				concept_dict_arr = get_abstract_concept_arr_dict(concept_df)
				abstract_concept_arr.extend(concept_dict_arr)
	except:
		return None
	if len(abstract_concept_arr) == 0:
		return None
	else:
		return abstract_concept_arr

if __name__ == "__main__":
	update_postgres_document_concept_count()