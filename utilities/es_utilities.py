from elasticsearch import Elasticsearch, RequestsHttpConnection
import os

def return_es_host():
	return {'host' : os.environ['ES_HOST'], 'port' : os.environ['ES_PORT'], 'request_timeout' : 1000}


def get_es_client():
	return Elasticsearch([return_es_host()])


def search(es_query, index_name):
	es = get_es_client()
	res = es.search(index=index_name, body=es_query)
	return res
