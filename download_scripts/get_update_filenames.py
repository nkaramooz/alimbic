import sys
from ftplib import FTP
sys.path.append('../utilities')
import pglib as pg
from bs4 import BeautifulSoup
import regex as re
import sqlalchemy as sqla


def get_new_update_files():
	conn,cursor = pg.return_postgres_cursor()
	conn, cursor = pg.return_postgres_cursor()
	query = """
		select max(file_num) from pubmed.indexed_files
	"""
	cursor.execute(query)
	max_num = cursor.fetchone()[0]
	cursor.close()
	conn.close()
	f = FTP('ftp.ncbi.nlm.nih.gov')
	f.login()
	f.cwd('/pubmed/updatefiles/')
	filenames = f.nlst()

	download_file_names = []
	for i in filenames:
		if re.match(".*\.xml\.gz$", i):
			file_num = int(re.findall('pubmed20n(.*)\.xml.gz$', i)[0])
			if file_num > max_num:
				download_file_names.append(i)
	return download_file_names


if __name__ == "__main__":
	print(' '.join(get_new_update_files()))

# resp = h_conn.getresponse()
# data = resp.read()
# soup = BeautifulSoup(data, 'html.parser')
# items = soup.find_all('tr')
# print(soup)
# print(data)
# print(items)



# print(5)
# sys.exit(0)