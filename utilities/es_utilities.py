from elasticsearch import Elasticsearch, RequestsHttpConnection


# Create your views here.

INDEX_NAME='pubmedx1.6'

def return_es_host():
	# return {'host': 'vpc-elasticsearch-ilhv667743yj3goar2xvtbyriq.us-west-2.es.amazonaws.com', 'port' : 443}
	return {'host' : 'localhost', 'port' : 9200, 'request_timeout' : 100000}

def get_es_client():
	# es = Elasticsearch(hosts=[return_es_host()], use_ssl=True, verify_certs=True, connection_class=RequestsHttpConnection)
	es = Elasticsearch([return_es_host()])
	return es



class ElasticScroll():
	def __init__(self,client, query):
		self.es = client
		self.initialized = False
		self.sid = None
		self.scroll_size = None
		self.has_next = True
		self.query = query
		self.counter = 0

	def next(self):
		if not self.initialized:
			pages = self.es.search(index=INDEX_NAME, scroll='5m', \
				size=500, request_timeout=100000, body={"query" : self.query})
			self.sid = pages['_scroll_id']
			self.scroll_size = pages['hits']['total']['value']
			self.initialized = True
			self.counter += 1
			return pages
		else:
			if self.scroll_size > 0:
				pages = self.es.scroll(scroll_id = self.sid, scroll='5m', request_timeout=100000)
				self.sid = pages['_scroll_id']
				self.scroll_size = len(pages['hits']['hits'])
				self.counter += 1
				if self.scroll_size == 0:
					self.has_next = False
				# if self.counter > 0:
				# 	self.has_next = False
				return pages

