import sys
sys.path.append('../../utilities')
sys.path.append('../..')
import http.client
import pandas as pd
import snomed_annotator2 as ann2
from bs4 import BeautifulSoup
import utils2 as u
import pglib as pg
import sqlalchemy as sqla
import urllib

# conn, cursor = pg.return_postgres_cursor()
# query = "select org_acr, url from guidelines.guideline_orgs where org_acr='NCCN'"

# orgs = pg.return_df_from_query(cursor, query, None, ["org_acr", "url"])

# url = orgs['url'][0]

s = urllib.request.urlopen('https://www.nccn.org/guidelines/category_1')
mybytes = s.read()
mystr = mybytes.decode("utf8")
s.close()
soup = BeautifulSoup(mystr, 'html.parser')
print(soup)
# h_conn = http.client.HTTPSConnection('www.nccn.org/guidelines/category_1')
# h_conn.request("GET", "/")

# resp = h_conn.getresponse()
# data = resp.read()
# soup = BeautifulSoup(data, 'html.parser')
# print(soup)