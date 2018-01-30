from django.conf.urls import url
from django.views.generic.base import RedirectView


from . import views

app_name = 'search'
urlpatterns = [
    url(r'^$', views.home_page),
    url(r'^elastic_search_home_page/$', views.elastic_search_home_page, name='elastic_search_home_page'),
    url(r'^elastic_search/$', views.post_elastic_search, name='elastic_search'),
    url(r'^elastic_search/(?P<query>.*)/$', views.elastic_search_results, name='elastic_search_results'),
    
    url(r'^concept_search_home_page/$', views.concept_search_home_page, name='concept_search_home_page'),
    # url(r'^concept_search/$', views.post_concept_search, name='concept_search'),
    url(r'^concept_search/pivot/$', views.post_pivot_search, name='pivot_search'),
    url(r'^concept_search/pivot_results/(?P<query>.*)/(?P<conceptid1>.*)/(?P<conceptid2>.*)/$', views.conceptid_search_results, name='conceptid_search_results'),
    # url(r'^concept_search/(?P<query>.*)/$', views.concept_search_results,  name='concept_search_results'),
    # url(r'^concept_search/(?P<query>.*)/(?P<journals>.*)/(?P<start_year>.*)/(?P<end_year>.*)/$', views.concept_search_results,  name='concept_search_results'),
    # url(r'^concept_search/(?P<params>.*)/$', views.concept_search_results,  name='concept_search_results'),
    url(r'^concept_search/$', views.concept_search_results,  name='concept_search_results'),
    url(r'^concept_override/post/$', views.post_concept_override, name='post_concept_override'),
    url(r'^concept_override/', views.concept_override, name='concept_override'),
    
]