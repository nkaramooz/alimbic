from django.urls import re_path
from django.views.generic.base import RedirectView

from . import views

app_name = 'search'
urlpatterns = [
    re_path(r'^about/', views.about, name='about'),
    re_path(r'^terms/', views.terms, name='terms'),
    re_path(r'^$', views.home, name='home'),
    re_path(r'^search/', views.post_search_text, name='post_search_text'),
]