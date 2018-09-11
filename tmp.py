c_df = acronym_check(res)

		#test reconstruction
		c_res = pd.DataFrame()
		sentence_arr = []

		line = clean_text(query44)
		ln_words = line.split()
		concept_counter = 0
		concept_len = len(c_df)
		at_end = False
		for ind1, word in enumerate(ln_words):
			print(word)
			added = False
			while (concept_counter < concept_len):
				if ((ind1 >= c_df.iloc[concept_counter]['description_start_index']) and (ind1 <= c_df.iloc[concept_counter]['description_end_index'])):
					sentence_arr.append((word, c_df.iloc[concept_counter]['conceptid']))
					added = True
					break
				else:
					sentence_arr.append((word, 0))
					added = True
					break

				if ((ind1 == c_df.iloc[concept_counter]['description_end_index'])):
					concept_counter +=1

			if not added:
				sentence_arr.append((word, 0))
		u.pprint(sentence_arr)