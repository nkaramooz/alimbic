# These test cases evaluate the snomed_annotator.py module.
# Column 3 of the queryArr is the expected output of the test case in
# the form of snomed codes. The test cases are not written in terms of the alimbic 
# concept ID since rebuilding the graph on a new machine will result in different
# alimbic concept ids (acids).

import unittest
import pandas as pd
import sys
import os
from snomed_annotator.snomed_annotator2 import annotate_text, get_cache, clean_text, get_all_words_list
from utilities.pglib import get_cid_from_acid
from nltk.stem.wordnet import WordNetLemmatizer

class TestAnnotator(unittest.TestCase):
	def test_equal(self):
		queryArr = [
			("protein C deficiency protein S deficiency", 100, 'complete', ['76407009', '1563006'])
			,("chronic obstructive pulmonary disease and congestive heart failure", 100, 'partial', ['13645005', '42343007'])
			,("chronic obstructive pulmonary disease and congestive heart failure", 80,'partial', ['13645005', '42343007'])
			,("protein C deficiency protein S deficiency", 80,'complete', ['76407009', '1563006'])
			,("ankyalosing spondylitis", 80,'complete', ['9631008'])
			,("Weekly vs. Every-3-Week Paclitaxel and Carboplatin for Ovarian Cancer", 100, 'partial', ['387374002', '386905002', '363443007'])
			,("Weekly vs. Every-3-Week Paclitaxel and Carboplatin for Ovarian Cancier", 80, 'partial', ['387374002', '386905002', '363443007'])
			,("Sustained recovery of progressive multifocal leukoencephalopathy after treatment with IL-2.", 100, 'partial', ['22255007', '68945006'])
			,("luteinizing hormone releasing hormone in thyroid hormone deficiency", 100, 'partial', ['49869009', '18220004'])
			,("Sacubitril/valsartan for congestive heart failure", 100, 'partial', ['777480008', '42343007'])
			,("Chronic obstructive pulmonary disease or COPD", 100, 'complete', ['13645005'])
			]

		lmtzr = WordNetLemmatizer()

		print("============================")
		for q in queryArr:
			print(q)
			term = q[0]
			term = clean_text(term)
			spellcheck_threshold = q[1]
			all_words = get_all_words_list(term)

			cache = get_cache(all_words_list=all_words, case_sensitive=True, \
				check_pos=False, spellcheck_threshold=spellcheck_threshold, lmtzr=lmtzr)
			sentences_df = pd.DataFrame([[term, 'title', 0,0]], \
				columns=['line', 'section', 'section_ind', 'ln_num'])
			item = pd.DataFrame([[term, 'title', 0, 0]], columns=['line', 'section', 'section_ind', 'ln_num'])

			res, g, s = annotate_text(sentences_df=sentences_df, cache=cache, \
				case_sensitive=True, check_pos=False, acr_check=False, \
				spellcheck_threshold=spellcheck_threshold, \
				write_sentences=True, lmtzr=None)	
			cids = [get_cid_from_acid(acid) for acid in s['concept_arr'].values[0]]
			if q[2] == 'complete':
				self.assertTrue(set(cids) == set(q[3]))
			else:
				print(cids)
				self.assertTrue(set.intersection(set(q[3]), set(cids))==set(q[3]))

		print("============================")

if __name__ == "__main__":
	unittest.main()
