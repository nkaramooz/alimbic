import spacy
from spacy.util import minibatch, compounding
import random
import sys
import pandas as pd
sys.path.append('../utilities')
import pglib as pg
sys.path.append('../')
import ml2 as ml

label = "DRUG"
example = [["apolipoprotein", "85156"], ["B100", "85156"], ["level", "839"], ["increase", "694075"], ["during", 0], ["asparaginase", "45220"], ["therapy", "45220"], ["although", 0], ["the", 0], ["mechanism", "226866"], ["of", "180698"], ["this", 0], ["remain", 0], ["unclear", 0]]


def gen_dataset_worker(input, all_treatments):
	for func,args in iter(input.get, 'STOP'):
		gen_dataset_calculate(func, args, all_treatments)


def gen_dataset_calculate(func, args, all_treatments):
	func(*args, all_treatments)


def gen_ner_data_top():
	conn,cursor = pg.return_postgres_cursor()
	all_treatments = get_all_treatments_set()
	query = "select min(ver) from spacy.drug_ner_sentences"
	new_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])+1
	old_version = new_version-1
	curr_version = old_version

	while curr_version != new_version:
		gen_ner_data_bottom(old_version, new_version, all_treatments)
		curr_version = int(pg.return_df_from_query(cursor, query, None, ['ver'])['ver'][0])


		
def gen_ner_data_bottom(old_version, new_version, all_treatments):
	number_of_processes = 1

	task_queue = mp.Queue()
	pool = []

	for i in range(number_of_processes):
		p = mp.Process(target=gen_dataset_worker, args=(task_queue,))
		pool.append(p)
		p.start()

	conn,cursor = pg.return_postgres_cursor()

	get_query = """
		select 
			sentence_id
			,sentence_tuples
		from spacy.drug_ner_sentences
		where ver = %s limit 1000
	"""

	ner_sentences_df = pg.return_df_from_query(cursor, get_query, \
				(old_version,), ['sentence_id', 'sentence_tuples'])

	counter = 0

	while (counter < number_of_processes) and (len(ner_sentences_df.index) > 0):
		params = (ner_sentences_df,all_treatments)
		task_queue.put((gen_drug_ner_dataset, params))
		update_query = """
			UPDATE spacy.drug_ner_sentences
			SET ver = %s
			where sentence_id = ANY(%s);
		"""
		cursor.execute(update_query, (new_version, ner_sentences_df['sentence_id'].values.tolist(), ))
		cursor.connection.commit()

		ner_sentences_df = pg.return_df_from_query(cursor, get_query, \
				(old_version,), ['sentence_id', 'sentence_tuples'])

		counter += 1

	for i in range(number_of_processes):
		task_queue.put('STOP')

	for p in pool:
		p.join()

	for p in pool:		
		p.close()

	cursor.close()
	conn.close()

def gen_drug_ner_dataset(ner_sentences_df, all_treatments):

	ner_sentences_dict = ner_sentences_df.to_dict('records')
	res_df = pd.DataFrame()
	for row in ner_sentences_dict:
		sentence_tuples = row['sentence_tuples']
		start_index = 0
		full_sentence = ""
		res_tuple = None
		res_dict = {"entities" : []}
		for item in sentence_tuples:
			full_sentence += item[0]
			full_sentence += " "
			if item[1] in all_treatments:
				end_index = start_index + len(item[0])
				res_dict["entities"].append((start_index, end_index, label))
			start_index += len(item[0])
		full_sentence = full_sentence.rstrip()
		res_tuple = (full_sentence, res_dict)
		res_df = res_df.append(pd.DataFrame([[row['sentence_id'], res_tuple, 0]], columns=['sentence_id', 'train_sentence', 'ver']))

	engine = pg.return_sql_alchemy_engine()

	res_df.to_sql('drug_ner_train', engine, schema='spacy', if_exists='append', \
		 index=False, dtype={'train_sentences' : sqla.types.JSON})

	cursor.close()
	conn.close()
	engine.dispose()

# for row in enumerate(ner_sentences_df):
# 	print(row)
# nlp = spacy.load("en_core_web_sm")
# ner = nlp.get_pipe('ner')

# ner.add_label(label)
# optimizer = nlp.resume_training()
# move_names = list(ner.move_names)
# pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
# other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# with nlp.disable_pipes(*other_pipes):
# 	sizes = compounding(1.0, 4.0, 1.001)
# 	for itn in range(2):
# 		batches = minibatch()