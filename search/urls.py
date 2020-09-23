from django.conf.urls import url
from django.views.generic.base import RedirectView


from . import views

app_name = 'search'
urlpatterns = [
    url(r'^$', views.concept_search_home_page, name='concept_search_home_page'),
    # url(r'^search/', views.concept_search_home_page, name='concept_search_home_page'),
    # url(r'^pivot/$', views.post_pivot_search, name='pivot_search'),

    # url(r'^results/$', views.concept_search_results,  name='concept_search_results'),
    url(r'^concept_override/post/$', views.post_concept_override, name='post_concept_override'),
    url(r'^concept_override/', views.concept_override, name='concept_override'),
    url(r'^training/post/$', views.post_training, name='post_training'),
    url(r'^training/', views.training, name='training'),
    url(r'^search/', views.post_search_text, name='post_search_text'),

    url(r'^ml/post/$', views.post_ml, name='post_ml'),
    url(r'^ml/', views.ml, name='ml'),


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
    # url(r'^ajax/getJournals/$', views.getJournals, name='getJournals'),
]