import sys
import io
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten
from keras.models import load_model
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import snomed_annotator as ann
from numpy import array 
from keras.layers import GlobalMaxPooling1D
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import utilities.utils as u, utilities.pglib as pg

model = load_model('model-01.hdf5')
print(model.layers)
emb = model.layers[0].get_weights()
print(emb)
print(len(emb[0]))
print(len(emb[0][1]))
print(emb[0][0][0])
# sys.exit(0)
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')

counter = 0
while counter < 10000:
	substring = emb[0][counter]
	for item in substring:
		item = str(item)
		# print(item)
		# sys.exit(0)
		out_v.write(str(item))
		out_v.write("\t")
	out_v.write("\n")
	counter += 1


UNK_ind=1
def get_word_from_ind(index):
	if index == UNK_ind:
		return 'UNK'
	else:
		conn,cursor = pg.return_postgres_cursor()
		query = "select word from ml2.word_counts_50k where rn = %s limit 1"
		word_df = pg.return_df_from_query(cursor, query, (index,), ['word'])
		return str(word_df['word'][0])

out_m = io.open('meta.tsv', 'w', encoding='utf-8')
words_list = ['padding', 'UNK']

conn,cursor = pg.return_postgres_cursor()
query = "select word from ml2.word_counts_50k where rn <10000"
word_df = pg.return_df_from_query(cursor, query, None, ['word'])
words_list.extend(word_df['word'].tolist())



for i in words_list:
	if i.isdigit():
		query = """
			select term from annotation2.preferred_concept_names
			where acid =%s
		"""
		term = pg.return_df_from_query(cursor, query, (i,), ['term'])
		
		if i=='421463005':
			print(term)

		if len(term) > 0:
			term = term['term'][0]
			out_m.write(term + "\n")
		else:
			out_m.write(i + "\n")
	else:
		out_m.write(i + "\n")



