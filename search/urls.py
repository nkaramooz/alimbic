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
    # url(r'^concept_search/(?P<query>.*)/$', views.concept_search_results,  name='concept_search_results'),
    # url(r'^concept_search/(?P<query>.*)/(?P<journals>.*)/(?P<start_year>.*)/(?P<end_year>.*)/$', views.concept_search_results,  name='concept_search_results'),
    # url(r'^concept_search/(?P<params>.*)/$', views.concept_search_results,  name='concept_search_results'),
    url(r'^concept_search/$', views.concept_search_results,  name='concept_search_results'),
    url(r'^concept_override/post/$', views.post_concept_override, name='post_concept_override'),
    url(r'^concept_override/', views.concept_override, name='concept_override'),

    url(r'^vancomycin/$', views.vc_main, name='vc_main'),
    url(r'^vancomycin/cases/(?P<uid>.*)$', views.vc_cases, name='vc_cases'),
    url(r'^vancomycin/case_view/(?P<cid>.*)$', views.vc_case_view, name='vc_case_view'),
    url(r'^vancomycin/case/(?P<cid>.*)/custom_dose$', views.custom_dose_form, name='custom_dose_form'),
    url(r'^vancomycin/case/(?P<cid>.*)/loading/form$', views.loading_form, name='loading_form'),
    url(r'^vancomycin/case/(?P<cid>.*)/loading_rec$', views.loading_rec, name='loading_rec'),
    url(r'^vancomycin/(?P<uid>.*)/new_case/$', views.vc_new_case, name='vc_new_case'),

    url(r'^vancomycin/case/(?P<cid>.*)/maintenance/form$', views.maintenance_form, name='maintenance_form'),
    url(r'^vancomycin/case/(?P<cid>.*)/maintenance/rec$', views.maintenance_rec, name='maintenance_rec'),

    url(r'^vancomycin/case/(?P<cid>.*)/redose/form$', views.redose_form, name='redose_form'),
    url(r'^vancomycin/case/(?P<cid>.*)/redose/rec$', views.redose_rec, name='redose_rec'),
    
    url(r'^ajax/vcTroughTarget/$', views.returnTroughTarget, name='returnTroughTarget'),
]