from django.urls import re_path
from django.views.generic.base import RedirectView


from . import views

app_name = 'graph'
urlpatterns = [
    re_path(r'^graph/', views.graph, name='graph'),
    re_path(r'^get_acid/', views.get_acid, name='get_acid'),
    re_path(r'^get_adid/', views.get_adid, name='get_adid'),
    re_path(r'^get_term/', views.get_term, name='get_term'),
    re_path(r'^post_label_rel/', views.post_label_rel, name='post_label_rel'),
    re_path(r'^get_rel/', views.get_rel, name='get_rel'),
    re_path(r'^create_concept/', views.create_concept, name='create_concept'),
    re_path(r'^deactivate_concept/', views.deactivate_concept, name='deactivate_concept'),
    re_path(r'^new_description/', views.new_description, name='deactivate_concept'),
    re_path(r'^deactivate_description/', views.deactivate_description, name='deactivate_description'),
    re_path(r'^modify_parent/', views.modify_parent, name='modify_parent'),
    re_path(r'^set_acronym/', views.set_acronym, name='set_acronym'),
    re_path(r'^set_concept_type/', views.set_concept_type, name='set_concept_type')
]