from django import template
from datetime import date, timedelta

register = template.Library()

@register.filter(name='get')
def get(dict, key):
	return dict.get(key, None)