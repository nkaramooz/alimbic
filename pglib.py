import psycopg2
import pandas as pd
from sqlalchemy import create_engine

def return_postgres_cursor():
	conn_string = "host='localhost' dbname='laso' user='LilNimster'"
	conn = psycopg2.connect(conn_string)
	cursor = conn.cursor()
	return cursor

def return_df_from_query(sql_query, column_names):
	cursor = return_postgres_cursor()
	cursor.execute(sql_query)
	records = cursor.fetchall()
	return pd.DataFrame(records, columns = column_names)

def return_sql_alchemy_engine():
	engine = create_engine('postgresql://LilNimster@localhost:5432/laso')
	return engine
