from elasticsearch import Elasticsearch, RequestsHttpConnection


# Create your views here.

INDEX_NAME='pubmedx2.0'

def return_es_host():
	# return {'host': 'vpc-elasticsearch-ilhv667743yj3goar2xvtbyriq.us-west-2.es.amazonaws.com', 'port' : 443}
	return {'host' : 'localhost', 'port' : 9200, 'request_timeout' : 100000}

def get_es_client():
	# es = Elasticsearch(hosts=[return_es_host()], use_ssl=True, verify_certs=True, connection_class=RequestsHttpConnection)
	es = Elasticsearch([return_es_host()])
	return es


