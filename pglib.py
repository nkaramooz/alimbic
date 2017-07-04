import psycopg2
import pandas as pd

def return_postgres_cursor():
	conn_string = "host='localhost' dbname='snomed' user='LilNimster'"
	conn = psycopg2.connect(conn_string)
	cursor = conn.cursor()
	return cursor

def return_df_from_query(sql_query):
	cursor = return_postgres_cursor()
	cursor.execute(sql_query)
	records = cursor.fetchall()
	return pd.DataFrame(records, columns = ['conceptid', 'term'])

