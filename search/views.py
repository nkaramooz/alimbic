from django.shortcuts import render

from elasticsearch import Elasticsearch
# Create your views here.

def home_page(request):
	return render(request, 'search/home_page.html') #this points to the HTML file

def search(request):
	es = Elasticsearch([{'host' : 'localhost', 'port' : 9200}])
	query = request.POST['query']
	es_query = {"from" : 0, "size" : 20, "query" : {"match": {'_all' : query}}}
	sr = es.search(index='pubmed', body=es_query)['hits']['hits']

	return render(request, 'search/results_page.html', {'sr' : sr})
