import xml.etree.ElementTree as ET
import pandas as pd
from bs4 import BeautifulSoup
import re
import pickle
import sys
import psycopg2 
import pglib as pg #import return_df_from_query, return_sql_alchemy_engine
from fuzzywuzzy import fuzz
import uuid
import codecs
import snomed_annotator as snomed
import sys
import nltk.data
import time
import utils as u
import multiprocessing as mp
from functools import partial


filterWordsFilename = 'filter_words'

def clean_text(text):
	result_string = ""

	for line in text.split('\n'):
		line = line.lower()

		try:
			if ('==external links==' == line) or ('==references==' == line):
				break
			elif bool(re.search('==.*==', line)):
				continue
			elif line[0:2] == '{|':
				continue
			elif line[0:2] == ' |':
				continue
			elif line[0] == '|':
				continue
			elif line[0:9] == 'ref name=':
				continue
			else:
				line = re.sub('{{.*}}', '', line)
				line = re.sub('{{.*', '', line)
				line = re.sub('.*}}', '', line)
				line = re.sub('\[\[category:.*\]\]', '', line)
				line = re.sub('\[\[image:.*\]\]', '', line)
				line = line.replace('[[', '')
				line = line.replace(']]', '')
				line = line.replace(',', '')
				# line = line.replace('.', '') ## commented out for document storage script
				line = line.replace('\'', '')
				line = line.replace('\"', '')
				line = line.replace('(', '')
				line = line.replace(')', '')
				if line != '':
					result_string = result_string + '\n' + line
		except:
			continue
	return result_string

def token_counts_frequency(text, docid, doc_title):

	count_df = pd.DataFrame(columns=('docid', 'doc_title', 'words', 'count'))

	words = text.split()

	for word in words:
		count_df.loc[len(count_df)] = [docid, doc_title, word, 1]

	count_df = count_df.groupby(['docid', 'doc_title', 'words']).sum().reset_index()

	return get_document_frequency_distribution(count_df)


def get_document_frequency_distribution(count_df):
	count_df['frequency'] = count_df['count'] / count_df['count'].sum()

	return count_df


def load_wikis():
	tree = ET.parse('articles.xml')

	root = tree.getroot()

	ns = {'file_ns': 'http://www.mediawiki.org/xml/export-0.10/'}

	final_results_df = pd.DataFrame(columns=('docid', 'doc_title', 'words', 'count'))

	for page in root.findall('file_ns:page', ns):
		docid = ""
		doc_title = ""

		for title in page.findall('file_ns:title', ns):
			doc_title = title.text

		for rev in page.findall('file_ns:revision', ns):
			
			for d_id in rev.findall('file_ns:id', ns):
				docid = d_id.text	

			for body in rev.findall('file_ns:text', ns):
				text = body.text
				soup = BeautifulSoup(text, 'html.parser')
				for tag in soup.find_all('strong'):
					tag.replaceWith('')

				text = soup.get_text()

				text = clean_text(text)
				

				document_df = token_counts_frequency(text, docid, doc_title)

				final_results_df = final_results_df.append(document_df)

		
	final_results_df.to_pickle('document_word_frequencies')

def save_clean_text():
	tree = ET.parse('articles.xml')

	root = tree.getroot()

	ns = {'file_ns': 'http://www.mediawiki.org/xml/export-0.10/'}

	file_system = pd.DataFrame(columns=['filename', 'docid'])


	for page in root.findall('file_ns:page', ns):
		docid = ""
		doc_title = ""

		for title in page.findall('file_ns:title', ns):
			doc_title = title.text

		for rev in page.findall('file_ns:revision', ns):
			
			for d_id in rev.findall('file_ns:id', ns):
				docid = d_id.text	

			for body in rev.findall('file_ns:text', ns):
				text = body.text
				soup = BeautifulSoup(text, 'html.parser')
				for tag in soup.find_all('strong'):
					tag.replaceWith('')

				text = soup.get_text()

				text = clean_text(text)
				
				filename = str(uuid.uuid4()) + ".txt"
				new_file = pd.DataFrame([[filename, docid]], columns=['filename', 'docid'])
				file_system = file_system.append(new_file)

				filename = "/Users/LilNimster/Documents/wiki_data/text_files/" + filename

				new_file = codecs.open(filename, 'w', encoding='utf8')
				new_file.write(text)
				new_file.close()


	file_system.to_pickle('file_system')
				

		

def aggregate_data():
	document_df = pd.read_pickle('document_word_frequencies')
	global_df = document_df[['words', 'count']]
	global_df = global_df.groupby(['words']).sum().reset_index()
	global_df['frequency'] = global_df['count'] / global_df['count'].sum()
	global_df = global_df.sort_values('frequency', ascending=False)
	global_df.to_pickle('global_word_frequencies')


def determine_filter_words():
	global_df = pd.read_pickle('global_word_frequencies')
	std = global_df['frequency'].std()
	mean = global_df['frequency'].mean()

	filter_words = global_df[global_df['frequency'] >= (4*std + mean)]
	filter_words.to_pickle(get_filter_words_filename())

def get_filter_words_filename():
	global filterWordsFilename
	return filterWordsFilename


def annotate_doc_store_with_snomed_parallel():

	file_system_df = pd.read_pickle('file_system')

	results_df = pd.DataFrame()
	pool = mp.Pool(processes=3)
	for doc_index, doc in file_system_df.iterrows():
		if doc['filename'] == 'e3b0949d-4549-43f0-b3b3-fdce04dca74b.txt':
			timer = u.Timer("10 sentences")
			file_path = "/Users/LilNimster/Documents/wiki_data/text_files/"
			file_path += doc['filename']

			current_doc = codecs.open(file_path, 'r', encoding='utf8')

			doc_text = current_doc.read()
			tokenized = nltk.sent_tokenize(doc_text)

			funclist = []
			counter = 0

			
			for ln_index, line in enumerate(tokenized):
				
				params = (line, ln_index, doc)
				funclist.append(pool.apply_async(annotate_doc, params))
				counter += 1


				if counter == 50:
					break
			
			pool.close()
			pool.join()

			for r in funclist:
				result = r.get()

				if result is not None:
					results_df = results_df.append(result)
			u.pprint(results_df)
			timer.stop()
			sys.exit(0)
			
	# engine = pg.return_sql_alchemy_engine()
	# results_df.to_sql('doc_annotation', engine, schema='annotation', if_exists='append')

def annotate_doc_store_with_snomed_parallel_v2():
	file_system_df = pd.read_pickle('file_system')

	results_df = pd.DataFrame()

	number_of_processes = 4

	for doc_index, doc in file_system_df.iterrows():
		if doc['filename'] == 'e3b0949d-4549-43f0-b3b3-fdce04dca74b.txt':
			timer = u.Timer("50 sentences")
			file_path = "/Users/LilNimster/Documents/wiki_data/text_files/"
			file_path += doc['filename']

			current_doc = codecs.open(file_path, 'r', encoding='utf8')

			doc_text = current_doc.read()
			tokenized = nltk.sent_tokenize(doc_text)

			funclist = []
			counter = 0

			task1 = []

			for ln_index, line in enumerate(tokenized):
				if counter == 25:
					line = line.encode('utf-8')
					line = line.replace('.', '')
					line = line.replace('!', '')
					line = line.replace(',', '')
					line = line.replace(';', '')
					line = line.replace('*', '')
					print(line)
					sys.exit(0)
					params = (line, ln_index, doc)
					task1.append((annotate_doc, params))
				
				counter +=1
				
				if counter > 25:
					break

			task_queue = mp.Queue()
			done_queue = mp.Queue()

			for task in task1:
				task_queue.put(task)

			for i in range(number_of_processes):
				mp.Process(target=worker, args=(task_queue, done_queue)).start()

			for i in range(len(task1)):
				results_df = results_df.append(done_queue.get())

			for i in range(number_of_processes):
				task_queue.put('STOP')
			# u.pprint(results_df)
			timer.stop()
			sys.exit(0)


def worker(input, output):
	for func, args in iter(input.get, 'STOP'):
		result = calculate(func, args)
		output.put(result)

def calculate(func, args):
	result = func(*args)
	return result


def annotate_doc(line, ln_index, doc):

	cursor = pg.return_postgres_cursor()
	line = line.encode('utf-8')
	line = line.replace('.', '')
	line = line.replace('!', '')
	line = line.replace(',', '')
	line = line.replace(';', '')
	line = line.replace('*', '')

	annotation = snomed.return_line_snomed_annotation(cursor, line, 100)

	if annotation is not None:
		annotation['docid'] = doc['docid']
		annotation['ln_number'] = ln_index
		return annotation
	else:
		return None

# def annotate_document_with_snomed(c):
# 	file_text = current_file.read()

if __name__ == "__main__":
	# document_system = pd.read_pickle('file_system')

	# for index, row in document_system.iterrows():
	# 	filename = row['filename']
	# 	filename = "/Users/LilNimster/Documents/wiki_data/text_files/" + filename
	# 	current_file = codecs.open(filename, 'r', encoding='utf8')
	# 	file_text = current_file.read()
	# 	print snomed.return_snomed_annotation(row['docid'], file_text)
	# 	break
	start_time = time.time()
	annotate_doc_store_with_snomed_parallel_v2()
	print("--- %s seconds ---" % (time.time() - start_time))
	# save_clean_text()


# def annotate_documents_with_snomed():


	#cursor.execute("select * from snomed.curr_description_f limit 10")
	#records = cursor.fetchall()

	# new_df = pd.DataFrame(records)
	# print new_df

	# pprint.pprint(records)



# print fuzz.ratio('chest pain', 'sprain')

# load_wikis()

# aggregate_data()

# print fuzz.ratio("this is a test", "this is a test!")

# return_filtered_query("coughs ins copd diseases")

# for ln_index,line in enumerate(tokenized
# 	# params = (line, ln_index)
# 	params = {'line': line, 'ln_index' : ln_index}
# 	star_annotate_doc(params)
# 	sys.exit(0)
# 	funclist.append(pool.map_async(star_annotate_doc, params))
# 	if counter == 10:
# 		break
# 	counter += 1
# results = []
# results.append(pool.map_async(star_annotate_doc, tokenized[0:10