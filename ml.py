import pandas as pd
import re
import psycopg2
from sqlalchemy import create_engine
from nltk.stem.wordnet import WordNetLemmatizer
from operator import itemgetter
import nltk.data
import numpy as np
import time
import utilities.utils as u, utilities.pglib as pg
import collections
import math
import os
import random
from tempfile import gettempdir
import zipfile
import sys
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import snomed_annotator as ann
import pickle as pk
from keras.datasets import imdb
import sys
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import load_model
import re
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

vocabulary_size = 10000

# need one for root (condition), need one for rel treatment
# need one for none (spacer)
vocabulary_spacer = 3
# [root, rel, spacer]

def get_word_counts():
	query = "select sentence_tuples from annotation.sentences2"
	conn,cursor = pg.return_postgres_cursor()
	sentence_df = pg.return_df_from_query(cursor, query, None, ['sentence_tuples'])
	sentences = sentence_df['sentence_tuples'].tolist()
	
	counts = {}

	for i,s in enumerate(sentences):
		s = re.findall('\(.*?\)', s)

		for words in s:
			words = words.strip('(')
			words = words.strip(')')
			words = tuple(words.split(","))


			if words[0] in counts.keys():
				counts[words[0]] = counts[words[0]] + 1
			else:
				counts[words[0]] = 1
	counts = collections.OrderedDict(sorted(counts.items(), key=itemgetter(1), reverse = True))

	with open('word_count.pickle', 'wb') as handle:
		pk.dump(counts, handle, protocol=pk.HIGHEST_PROTOCOL)

	return counts

# def read_data():
# 	query = "select sentence_tuples from annotation.sentences2 limit 10"
# 	conn,cursor = pg.return_postgres_cursor()
# 	sentence_df = pg.return_df_from_query(cursor, query, None, ['sentence_tuples'])
# 	sentences = sentence_df['sentence_tuples'].tolist()
# 	word_list = []

# 	for i,s in sentence_df.iterrows():
# 		print(s['sentence_tuples'])
# 		for words in s['sentence_tuples']:
# 			words = words.strip('(')
# 			words = words.strip(')')
# 			words = tuple(words.split(","))
			
# 			word_list.append(words[0])
	
# 	return word_list

# def build_dataset(words, n_words):
#   """Process raw inputs into a dataset."""
#   count = [['UNK', -1]]
#   count.extend(collections.Counter(words).most_common(n_words - 1))
#   dictionary = dict()
#   for word, _ in count:
#     dictionary[word] = len(dictionary)
#   data = list()
#   unk_count = 0
#   for word in words:
#     index = dictionary.get(word, 0)
#     if index == 0:  # dictionary['UNK']
#       unk_count += 1
#     data.append(index)
#   count[0][1] = unk_count
#   reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
#   print(reversed_dictionary)

#   with open('reversed_dictionary.pickle', 'wb') as rd:
#   	pk.dump(reversed_dictionary, rd, protocol=pk.HIGHEST_PROTOCOL)

#   with open('dictionary.pickle', 'wb') as di:
#   	pk.dump(dictionary, di, protocol=pk.HIGHEST_PROTOCOL)

#   return data, count, dictionary, reversed_dictionary


# def build():
# 	with open('word_count.pickle', 'rb') as handle:
# 		counts = pk.load(handle)
# 	data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size-vocabulary_spacer)
# 	del vocabulary


def load_word_counts_dict():
	with open('word_count.pickle', 'rb') as handle:
		counts = pk.load(handle)
		counter = 1
		final_dict = {}
		reverse_dict = {}
		final_dict['UNK'] = 0
		reverse_dict[0] = 'UNK'
		for item in counts.items():
			final_dict[item[0]] = counter
			reverse_dict[counter] = item[0]
			counter += 1
			if counter == (vocabulary_size-vocabulary_spacer):
				break

		with open('reversed_dictionary.pickle', 'wb') as rd:
			pk.dump(reverse_dict, rd, protocol=pk.HIGHEST_PROTOCOL)

		with open('dictionary.pickle', 'wb') as di:
			pk.dump(final_dict, di, protocol=pk.HIGHEST_PROTOCOL)
		# print(final_dict)
		# print(reverse_dict)
		# print(reverse_dict[1])

# How does training 
#[[X (i.e. condition), Y (i.e. treatment), label]]
# Will need to get all children of X, and all children of Y
train_set = [['22298006','387458008', 1], #aspirin
	['22298006', '33252009', 1], # beta blocker
	['22298006', '734579009', 1], # ace inhibitor
	['22298006', '29857009', 0],
	['22298006', '267036007', 0],
	['44054006', '7947003', 0],
	['44054006', '400556009', 1],
	['44054006', '418285008', 0],
	['44054006', '48698004', 0],
	['44054006', '7092007', 0],
	['44054006', '386879008', 1],
	['44054006', '63718003', 0],
	['44054006', '391858005', 0],
	['44054006', '387419003', 1],
	['44054006', '387124009', 0],
	['44054006', '308111008', 0],
	['44054006', '1182007', 0],
	['44054006', '308113006', 0],
	['44054006', '96302009', 0],
	['44054006', '102747008', 0],
	['44054006', '308111008', 0],
	['44054006', '44054006', 0],
	['44054006', '38341003', 0],
	['44054006', '230690007',0],
	['34000006', '68887009', 1],
	['34000006', '395726003', 1],
	['34000006', '386835005', 1],
	['34000006', '414805007', 1],
	['34000006', '372574004', 1],
	['34000006', '387248006', 1],
	['34000006', '79440004', 1],
	['34000006', '363779003', 0],
	['34000006', '271737000', 0],
	['34000006', '14189004', 0],
	['34000006', '24526004', 0],
	['25374005', '346712003', 0],
	['25374005', '398866008', 1],
	['25374005', '108418007 ', 1],
	['93880001', '119746007', 1],
	['93880001', '367336001', 1],
	['93880001', '261479009', 0],
	['363418001', '367336001', 1],
	['363418001', '386920008', 1],
	['363418001', '450556004', 0],
	['13645005', '79440004', 1],
	['13645005', '428311008', 1],
	['13645005', '59187003', 0],
	['13645005', '267036007', 0],
	['37796009', '395892000', 1],
	['37796009', '33635003', 0],
	['37796009', '37796009', 1],
	['37796009', '58732000', 0],
	['37796009', '44868003', 1],
	['37796009', '241671007', 0],
	['37796009', '16310003', 0],
	['37796009', '230690007', 0],
	['37796009', '398057008', 0],
	['74400008', '80146002', 1],
	['74400008', '255631004', 1],
	['74400008', '7947003', 0],
	['74400008', '367336001', 0],
	['74400008', '27658006', 1],
	['74400008', '16310003', 0],
	['74400008', '77477000', 0],
	['74400008', '74400008', 0],
	['64766004', '387501005', 1],
	['42343007', '439569004', 0],
	['42343007', '372787008', 1],
	['42343007', '15222008', 1],
	['42343007', '5924003', 1],
	['42343007', '108480007', 1],
	['42343007', '722048006', 1],
	['42343007', '407059007', 0],
	['42343007', '22298006', 0],
	['42343007', '29857009', 0],
	['42343007', '267036007', 0],
	['195967001', '406443008', 1],
	['195967001', '79440004', 1],
	['195967001', '6710000', 0],
	['195967001', '82918005', 0],
	['195967001', '46947000', 1],
	['68566005', '255631004', 1],
	['68566005', '49650001', 0],
	['68566005', '372840008', 0],
	['68566005', '68566005', 1],
	['68566005', '55741006', 1],
	['68566005', '418752001', 0],
	['68566005', '416838001', 0],
	['68566005', '61425002', 0],
	['68566005', '36689008', 0],
	['62315008', '346712003', 0],
	['62315008', '409585005', 1],
	['62315008', '82622003', 1],
	['29857009', '14816004', 0],
	['29857009', '225390008', 0],
	['29857009', '418272005', 0],
	['422587007', '367336001', 0],
	['422587007', '103735009', 0],
	['422587007', '56549003', 1],
	['422587007', '108418007', 1],
	['422587007', '48875009', 0],
	['422587007', '32955006', 1],
	['422587007', '796001', 0],
	['25064002', '108755008', 0],
	['24700007', '49327004', 1],
	['24700007', '108809004', 1],
	['24700007', '449000008', 1],
	['24700007', '20720000', 1],
	['24700007', '417486008', 0],
	['24700007', '386042006', 0],
	['279039007', '229559001', 1],
	['279039007', '372588000', 1],
	['51615001', '69236009', 0],
	['51615001', '387281007', 0],
	['129458007', '14816004', 0],
	['51615001', '6710000', 1],
	['51615001', '76591000', 0],
	['51615001', '77465005', 1],
	['19030005', '27479000', 1],
	['87522002', '3829006', 1],
	['31712002', '41143004', 1],
	['31712002', '18165001', 0],
	['31712002', '68887009', 1],
	['420054005', '53041004', 0],
	['266468003', '372772003', 1],
	['266468003', '77465005', 1],
	['266468003', '49722008', 1],
	['266468003', '18165001', 0],
	['235595009', '108665006', 1],
	['235595009', '359887003', 1],
	['235595009', '359890009', 1],
	['235595009', '108661002', 1],
	['40930008', '404836005', 0],
	['40930008', '73187006', 1],
	['40930008', '387220006', 0],
	['34486009', '14399003', 1],
	['34486009', '69236009', 0],
	['66999008', '4076007', 0],
	['66999008', '409392004', 1],
	['49436004', '64597002', 1],
	['49436004', '250980009', 1],
	['49436004', '796001', 1],
	['49436004', '48603004', 1],
	['49436004', '69236009', 1],
	['49436004', '108581009', 1],
	['49436004', '64432007', 0],
	['49436004', '33442008', 0],
	['49436004', '80313002', 0],
	['123799005', '418285008', 1],
	['38341003', '42605004', 0],
	['38341003', '321719003', 0],
	['38341003', '123799005', 0],
	['38341003', '385561008', 1],
	['3238004', '73133000', 1],
	['3238004', '34646007', 1],
	['3238004', '64597002', 0],
	['3238004', '58390007', 0],
	['3238004', '86977007', 0],
	['3238004', '85598007', 0],
	['3238004', '373945007', 0],
	['3238004', '29857009', 0],
	['55004003', '407033009', 0],
	['55004003', '77671006', 0],
	['69878008', '387166005', 1],
	['69878008', '31895006', 1],
	['69878008', '21825005', 1],
	['69878008', '25217009', 1],
	['69878008', '44868003', 1],
	['69878008', '108772001', 1],
	['69878008', '399939002', 0],
	['69878008', '414916001', 0],
	['69878008', '386911004', 1],
	['82525005', '3258003', 0],
	['233604007', '346712003', 0],
	['233604007', '255631004', 1],
	['233604007', '717778001', 1],
	['233604007', '13645005', 0],
	['67782005', '243157005', 1],
	['67782005', '79440004', 1],
	['67782005', '233573008', 1],
	['67782005', '408275002', 0],
	['67782005', '40232005', 1],
	['67782005', '233604007', 0],
	['370143000', '372720008', 1],
	['370143000', '372664007', 1],
	['370143000', '321958004', 1],
	['370143000', '349854005', 1],
	['370143000', '407033009', 1],
	['370143000', '372726002', 1],
	['370143000', '49722008', 0],
	['370143000', '443730003', 1],
	['370143000', '420507002', 0],
	['370143000', '74732009', 0],
	['197480006', '166001', 1],
	['197480006', '372720008', 1],
	['197480006', '372664007', 1],
	['197480006', '264603002', 1],
	['47505003', '321958004', 1],
	['58214004', '10784006', 1],
	['58214004', '108386000', 1],
	['63634009', '116099008', 1],
	['63634009', '409405006', 1],
	['63634009', '367336001', 1],
	['63634009', '350086004', 0],
	['63634009', '409403004', 1],
	['63634009', '387227009', 1],
	['38713004', '409073007', 0],
	['38713004', '367336001', 1],
	['50711007', '35063004', 1],
	['50711007', '49327004', 1],
	['50711007', '136111001', 1],
	['50711007', '421559001', 1],
	['50711007', '252162007', 0],
	['50711007', '387499002', 0],
	['50711007', '363779003', 0],
	['50711007', '266468003', 0],
	['50711007', '25370001', 0],
	['50711007', '271737000', 0],
	['25370001', '64597002', 1],
	['25370001', '65801008', 1],
	['25370001', '49327004', 0],
	['25370001', '77465005', 1],
	['25370001', '25370001', 0],
	['25370001', '373345002', 1],
	['25370001', '422528000', 0],
	['25370001', '6710000', 1],
	['91637004', '68887009', 1],
	['91637004', '350344000', 1],
	['91637004', '43450002', 1],
	['91637004', '109131004', 1],
	['6142004', '71181003', 1],
	['6142004', '372532009', 1],
	['6142004', '409228005', 1],
	['6142004', '387010007', 1],
	['65363002', '255631004', 1],
	['65363002', '417901007', 1],
	['65363002', '387021003', 1],
	['65363002', '119954001', 1],
	['65363002', '27658006', 1],
	['128350005', '255631004', 1],
	['2576002', '387531004', 1],
	['2576002', '414111007', 0],
	['49049000', '387086006', 1],
	['49049000', '59187003', 0],
	['49049000', '387039007', 1],
	['49049000', '372498008', 1],
	['49049000', '395770007', 1],
	['49049000', '26929004', 0],
	['202855006', '79440004', 1],
	['202855006', '229559001', 1],
	['202855006', '372588000', 1],
	['413444003', '14816004', 0],
	['38341003', '14816004', 0],
	['394659003', '14816004', 0],
	['230690007', '14816004', 0],
	['37796009', '25064002', 0],
	['68566005', '49650001', 0],
	['49436004', '80313002', 0],
	['49436004', '230690007', 0],
	['233604007', '49727002', 0],
	['44054006', '308111008', 0]]

test_set = [
	['38341003', '270954007', 0],
	['38341003', '230690007', 0],
	['709044004', '30326004', 0],
	['709044004', '15222008', 1],
	['709044004', '53304009', 0],
	['709044004', '77465005 ', 0],
	['709044004', '266700009', 0],
	['709044004', '398887003', 1],
	['709044004', '709044004', 0],
	['709044004', '302497006', 1]]

def get_data(labeled_set, filename):

	conn,cursor = pg.return_postgres_cursor()
	labelled_set = pd.DataFrame()
	results = pd.DataFrame()
	for index,item in enumerate(labeled_set):

		root_cids = [item[0]]
		root_cids.extend(ann.get_children(item[0], cursor))

		rel_cids = [item[1]]
		rel_cids.extend(ann.get_children(item[1], cursor))


		# Want to select sentenceIds that contain both
		sentence_query = "select sentence_tuples from annotation.sentences where conceptid in %s and id in (select id from annotation.sentences where conceptid in %s)"

		#this should give a full list for the training row
		sentences = pg.return_df_from_query(cursor, sentence_query, (tuple(root_cids), tuple(rel_cids)), ["sentence_tuples"])
		# Now you get the sentence, but you need to remove root and replace with word index
		# Create sentence list
		# each sentence list contains a list vector for word index
		label = item[2]
	
		for index,sentence in sentences.iterrows():

			sentence_root_index = None
			sentence_rel_index = None
			for index,words in enumerate(sentence['sentence_tuples']):

				words = words.strip('(')
				words = words.strip(')')
				words = tuple(words.split(","))	
				
				# Will only handle one occurrence of each conceptid for simplification
				if words[1] in root_cids:
					sentence_root_index = index
				elif words[1] in rel_cids:
					sentence_rel_index = index


				# while going through, see if conceptid associated then add to dataframe with root index and rel_type index
			results = results.append(pd.DataFrame([[sentence, sentence_root_index, sentence_rel_index, label]], columns=['sentence_tuples', 'root_index', 'rel_index', 'label']))
	print(len(results))
	results.to_pickle(filename)

def serialize_data(filename):
	sample_train = pd.read_pickle(filename)
	with open('reversed_dictionary.pickle', 'rb') as rd:
  		reverse_dictionary = pk.load(rd)

	  	with open('dictionary.pickle', 'rb') as d:
	  		dictionary = pk.load(d)
	

	max_ln_arr = []
	for index, sentence in sample_train.iterrows():
		for index, tup in sentence['sentence_tuples'].iteritems():
			max_ln_arr.append(len(tup))
	
	## this is how much should be padded
	max_words = max(max_ln_arr)
	max_words = 20
	print("max_ln: " + str(max_words))
	x_train = []
	y_train = []

	for index, sentence in sample_train.iterrows():
		root_index = sentence['root_index']
		rel_index = sentence['rel_index']
		root_cid = None
		rel_cid = None
		# unknowns will have integer of 50000
		sample = [(vocabulary_size-1)]*max_words

		label = sentence['label']
		for index, tup in sentence['sentence_tuples'].iteritems():
			for ind, t in enumerate(tup):

				t = t.lower()
	
				t = t.strip('(')
				t = t.strip(')')
				t = tuple(t.split(","))	


				if ind == root_index and root_cid == None:
					root_cid = t[1]
					sample[ind] = vocabulary_size-3
					# append index for root
				elif ind == rel_index and rel_cid == None:
					rel_cid = t[1]
					sample[ind] = vocabulary_size-2
					# append index for rel
				elif t[1] == root_cid:
					sample[ind] = vocabulary_size-2
				elif t[1] == rel_index:
					sample[ind] = vocabulary_size-2
				elif t[0] in dictionary.keys():
					sample[ind] = dictionary[t[0]]
				else:
					sample[ind] = dictionary['UNK']
				if ind >= max_words-1:
					break
		x_train.append(sample)
		y_train.append(label)

	# print(len(x_train))
	# print(len(y_train))
	return x_train, y_train, max_words


def build_model():
	x_train, y_train, max_words = serialize_data("./sample_train.pkl")

	x_test, y_test, max_words = serialize_data("./test_set.pkl")
	# x_train = sequence(x_train)
	x_train = np.array(x_train)
	y_train = np.array(y_train)

	x_test = np.array(x_test)
	y_test = np.array(y_test)
	
	# y_train = sequence(y_train)
	# print(x_train.shape)
	embedding_size=32
	model=Sequential()
	model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
	# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	# model.add(MaxPooling1D(pool_size=2))
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))

	print(model.summary())
	model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])
	batch_size = 32
	num_epochs = 5
	x_valid, y_valid = x_train[:batch_size], y_train[:batch_size]
	x_train2, y_train2 = x_train[batch_size:], y_train[batch_size:]

	print(x_train2.shape)
	print(y_train2.shape)
	
	model.fit(x_train2, y_train2, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

	print(x_test)
	scores = model.evaluate(x_test, y_test, verbose=1)
	print(scores)
	print('Test accuracy:', scores[1])

	# for i,d in enumerate(x_test):
	# 	print(model.predict(np.array([d])))
	# 	print(y_train[i])
	# 	print("=========" + str(i))
	# 	if i == 1000:
	# 		break
	# cd = np.array([x_test[1301]])
	# print(model.predict(cd))
	# print(y_train[1301])
	# scores = model.evaluate(steps=1)
			
			# word_list.append(words[0])
			# if words[1] in root then number is vocabulary_size-2
			# if words[1] in rel then number is vocabulary_size-1
			# if words[0] in dictionary, then number
			# else then position for UNK.
# 1) pull sentences with both concept types
# 2) figure out how to get word sequences within range that still will include both conceptids

# vector [1-50k, one for UNK, one for key, one for value]

# get_data(train_set, './sample_train.pkl')
# get_data(test_set, './test_set.pkl')
# load_word_counts_dict()
build_model()
