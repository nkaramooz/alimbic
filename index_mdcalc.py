import http.client
import pandas as pd
import sys
import snomed_annotator2 as ann2
import utilities.pglib as pg
from bs4 import BeautifulSoup
import utilities.utils2 as u
import sqlalchemy as sqla

conn,cursor = pg.return_postgres_cursor()
h_conn = http.client.HTTPSConnection("www.mdcalc.com")
h_conn.request("GET", "/")

resp = h_conn.getresponse()
data = resp.read()
soup = BeautifulSoup(data, 'html.parser')
calc_items = soup.find_all('li', class_='calc-item')
print("number of calculators: " + str(len(calc_items)))

res_df = pd.DataFrame()
counter = 0
for ind,item in enumerate(calc_items):
	counter += 1
	# print(item.prettify())

	title_class = None
	desc_class = None
	if item.find_all(class_="calc-item__name index_most-popular_calcItem"):
		title_class = "calc-item__name index_most-popular_calcItem"
		desc_class = "calc-item__desc index_most-popular_calcItem"

	elif item.find_all(class_="calc-item__name index_all_calcItem"):
		title_class = "calc-item__name index_all_calcItem"
		desc_class = "calc-item__desc index_all_calcItem"

	elif item.find_all(class_="calc-item__name index_newest_calcItem"):
		title_class = "calc-item__name index_newest_calcItem"
		desc_class = "calc-item__desc index_newest_calcItem"

	elif item.find_all(class_="calc-item__name index_guidelines_calcItem"):
		title_class = "calc-item__name index_guidelines_calcItem"
		desc_class = "calc-item__desc index_guidelines_calcItem"

	elif item.find_all(class_="calc-item__name index_cmeCalcs_calcItem"):
		title_class = "calc-item__name index_cmeCalcs_calcItem"
		desc_class = "calc-item__desc index_cmeCalcs_calcItem"

	url = item.find_all('a', href=True)[0]['href']
	url = "www.mdcalc.com" + url
	title = item.find_all(class_=title_class)[0]

	title_text = title.get_text()
	title_text_clean = ann2.clean_text(title_text)
	title_all_words = ann2.get_all_words_list(title_text_clean)

	try:
		cache = ann2.get_cache(title_all_words, True)
		title_df = pd.DataFrame([[title_text_clean, 'title', 0, 0]], columns=['line', 'section', 'section_ind', 'ln_num'])
		title_annotation = ann2.annotate_text_not_parallel(title_df, cache, True, True, False)
		title_annotation = title_annotation[title_annotation['acid'].isnull() == False].copy()
	except:
		print(item)
	
	desc = item.find_all(class_=desc_class)[0]
	desc_text = desc.get_text()
	desc_text_clean = ann2.clean_text(desc_text)
	desc_all_words = ann2.get_all_words_list(desc_text_clean)
	cache = ann2.get_cache(desc_all_words, True)

	desc_df = pd.DataFrame([[desc_text_clean, 'title', 0, 0]], columns=['line', 'section', 'section_ind', 'ln_num'])
	desc_annotation = ann2.annotate_text_not_parallel(desc_df, cache, True, True, False)
	desc_annotation = desc_annotation[desc_annotation['acid'].isnull() == False].copy()
	final_annotation = title_annotation.append(desc_annotation)
	final_annotation = ann2.acronym_check(final_annotation)
	final_annotation = list(set(final_annotation['acid'].tolist()))
	final_annotation = pd.DataFrame(final_annotation, columns=['acid'])
	final_annotation['title'] = title_text
	final_annotation['desc'] = desc_text
	final_annotation['url'] = url

	res_df = res_df.append(final_annotation, sort=False)
print("number of calculators written: " + str(counter))
engine = pg.return_sql_alchemy_engine()

res_df.to_sql('mdc_staging', engine, schema='annotation', if_exists='replace', index=False, dtype={'title' : sqla.types.Text, 'desc' : sqla.types.Text, 'url' : sqla.types.Text})

engine.dispose()