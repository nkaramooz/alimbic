import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import time


def return_postgres_cursor():
	conn_string = "host='localhost' dbname='laso' user='LilNimster'"
	conn = psycopg2.connect(conn_string)
	cursor = conn.cursor()
	return cursor

def return_df_from_query(cursor, sql_query, params, column_names):
	cursor.execute(sql_query, params)
	records = cursor.fetchall()
	return pd.DataFrame(records, columns = column_names)

def return_sql_alchemy_engine():
	engine = create_engine('postgresql://LilNimster@localhost:5432/laso')
	return engine


class Timer:
	def __init__(self, label):
		self.label = label
		self.start_time = time.time()

	def stop(self):
		self.end_time = time.time()
		label = self.label + " : " + str(self.end_time - self.start_time)
		print(label)