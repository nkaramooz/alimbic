from django.conf.urls import url
from django.views.generic.base import RedirectView
from django.urls import path


from . import views

app_name = 'search'
urlpatterns = [
    url(r'^about/', views.about, name='about'),
    url(r'^terms/', views.terms, name='terms'),
    url(r'^$', views.home, name='home'),
    url(r'^search/', views.post_search_text, name='post_search_text'),
]