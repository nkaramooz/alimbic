from django.conf.urls import url
from django.views.generic.base import RedirectView


from . import views

app_name = 'graph'
urlpatterns = [
    url(r'^graph/post/$', views.post_concept_override, name='post_concept_override'),
    url(r'^graph/', views.graph, name='graph'),
    url(r'^get_acid/', views.get_acid, name='get_acid'),
    url(r'^get_adid/', views.get_adid, name='get_adid'),
    url(r'^get_term/', views.get_term, name='get_term'),
    url(r'^post_label_rel/', views.post_label_rel, name='post_label_rel'),
    url(r'^get_rel/', views.get_rel, name='get_rel'),
    url(r'^create_concept/', views.create_concept, name='create_concept'),
    url(r'^deactivate_concept/', views.deactivate_concept, name='deactivate_concept'),
    url(r'^new_description/', views.new_description, name='deactivate_concept'),
    url(r'^deactivate_description/', views.deactivate_description, name='deactivate_description'),
    url(r'^modify_parent/', views.modify_parent, name='modify_parent'),
    url(r'^set_acronym/', views.set_acronym, name='set_acronym'),
    url(r'^set_concept_type/', views.set_concept_type, name='set_concept_type')
]