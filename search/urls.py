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
    # url(r'^training/post/$', views.post_training, name='post_training'),
    # url(r'^training/', views.training, name='training'),
    url(r'^search/', views.post_search_text, name='post_search_text'),
    # url(r'^training/', views.training, name='training'),
    # url(r'^labelling/post/$', views.post_labelling, name='post_labelling'),
    # url(r'^labelling/', views.labelling, name='labelling'),

    # url(r'^ml/post/$', views.post_ml, name='post_ml'),
    # url(r'^ml/', views.ml, name='ml'),

    # url(r'^ajax/getJournals/$', views.getJournals, name='getJournals'),
]