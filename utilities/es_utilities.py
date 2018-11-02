import snomed_annotator as ann
import utilities.pglib as pg
from elasticsearch import Elasticsearch
from nltk.stem.wordnet import WordNetLemmatizer
import utilities.utils as u
import pandas as pd

# Create your views here.

INDEX_NAME='pubmedx1.2'

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

def is_child_of(child_id, parent_id, cursor):
	query = "select subtypeid from snomed.curr_transitive_closure_f where supertypeid=%s and subtypeid=%s"
	df = pg.return_df_from_query(cursor, query, (child_id, parent_id), ["subtypeid"])
	if len(df) > 0:
		return True 
	else:
		return False

def is_parent_of(parent_id, child_id, cursor):
	query = "select subtypeid from snomed.curr_transitive_closure_f where subtypeid=%s and supertypeid=%s"
	df = pg.return_df_from_query(cursor, query, (child_id, parent_id), ["subtypeid"])
	if len(df) > 0:
		return True 
	else:
		return False

def get_relation(conceptid_1, conceptid_2, cursor):
	if is_child_of(conceptid_1, conceptid_2, cursor):
		return "child"
	elif is_parent_of(conceptid_1, conceptid_2, cursor):
		return "parent"
	else:
		return None

class Node():
	def __init__(self, conceptid, label):
		self.label = label
		self.conceptid = conceptid
		self.children = list()
		self.parent = None

	def add_child(self, child_node):
		self.children.append(child_node)
	def remove_child(self, child_node):
		self.children.remove(child_node)

	def has_children(self):
		if len(self.children) > 0:
			return True
		else:
			return False


class NodeTree():
	def __init__(self):
		self.root_list = list()

	def get_root_nodes(self):
		return self.root_node

	def add(self, a_conceptid, a_label, cursor):
		
		self.internal_add_node(a_conceptid, a_label, self.root_list, None, cursor)

	def internal_add_node(self, a_conceptid, a_label, candidate_list, parent_node, cursor):

		if len(candidate_list) == 0:
			a_node = Node(a_conceptid, a_label)
			a_node.parent = parent_node
			candidate_list.append(a_node)
		else:
			completed = False
			for n in candidate_list:
				if is_child_of(a_conceptid, n.conceptid, cursor):
					child = n
					self.internal_add_node(a_conceptid, a_label, n.children, n, cursor)
					completed = True
					break
				elif is_parent_of(a_conceptid, n.conceptid, cursor):
					a_node = Node(a_conceptid, a_label)
					a_node.parent = parent_node
					if parent_node is not None:
						parent_node.add_child(a_node)
						parent_node.remove_child(n)
					completed = True
					break
			if not completed:
				a_node = Node(a_conceptid, a_label)
				a_node.parent = parent_node
				candidate_list.append(a_node)
				a_node.parent = parent_node





if __name__ == "__main__":
	update_postgres_document_concept_count()