
import utilities.es_utilities as es_util
INDEX_NAME='pubmedx1.9'

def get_text_query(query):
	es_query = {
			 "query": {"bool" : {"must" : 
			 {"query_string": {"query": query}}, "must_not" : [{'term': {'article_type': 'Letter'}}] 
			 		} }}
	return es_query


es = es_util.get_es_client()
es_query = get_text_query("heart failure")

sr = es.search(index=INDEX_NAME, body=es_query)
print(sr)