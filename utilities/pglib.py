import psycopg2
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import time
import os

DB_S = ''

def return_postgres_cursor():
	# conn_string = "host='laso-or.cmzwr3t5wsym.us-west-2.rds.amazonaws.com' dbname='laso' connect_timeout=5 user='laso_db' password='%s'" % DB_S
	conn_string = "host='localhost' dbname='laso' user='LilNimster' connect_timeout=50"
	# conn_string = "host='localhost' dbname='laso' user='Nima'"
	conn = psycopg2.connect(conn_string)
	cursor = conn.cursor()
	return conn,cursor

def return_postgres_conn():
	# conn_string = "host='laso-or.cmzwr3t5wsym.us-west-2.rds.amazonaws.com' dbname='laso' connect_timeout=5 user='laso_db' password='%s'" % DB_S
	conn_string = "host='localhost' dbname='laso' user='LilNimster'"
	conn = psycopg2.connect(conn_string)
	return conn

def return_df_from_query(cursor, sql_query, params, column_names):
	cursor.execute(sql_query, params)
	records = cursor.fetchall()
	return pd.DataFrame(records, columns = column_names)

def return_sql_alchemy_engine():
	# engine = create_engine('postgresql://laso_db:%s@laso-or.cmzwr3t5wsym.us-west-2.rds.amazonaws.com:5432/laso') % DB_S
	engine = create_engine('postgresql://LilNimster@localhost:5432/laso')
	# engine = create_engine('postgresql://Nima@localhost:5432/laso')
	return engine

def return_numpy_from_query(cursor, sql_query, params):
	cursor.execute(sql_query, params)
	records = np.array(cursor.fetchall())
	return records


class Timer:
	def __init__(self, label):
		self.label = label
		self.start_time = time.time()

	def stop(self):
		self.end_time = time.time()
		label = self.label + " : " + str(self.end_time - self.start_time)
		print(label)