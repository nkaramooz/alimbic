import pandas as pd
import pglib as pg
import utils as u

#queries the active concepts table
def pull_all_acronyms(cursor):
	cursor = pg.return_postgres_cursor()

	query = """

		select conceptid, term, candidate
		from (
			select conceptid, term, btrim((regexp_matches(term, '(.*?) - '))[1], ' ') as candidate
			from annotation.active_concept_descriptions
			) tb 
		where (candidate not in (select term from annotation.active_concept_descriptions))
	"""
	
	candidate_acronym_df = pg.return_df_from_query(cursor, query, (), ['conceptid', 'term', 'candidate'])

	return candidate_acronym_df

def de_dupe_and_filter(candidate_acronym_df, snomed_df):

	candidate_index = []
	# evaluate for complete overlap with existing snomed
	# if there is overlap, then the candidate is not added
	# to the list of recommendations
	for index, row in candidate_acronym_df.iterrows():
	
		if len(snomed_df[snomed_df['term'] == row['candidate'][0].strip()]) > 0:
			continue
		else:
			candidate_index.append(index)

	results = candidate_acronym_df[candidate_acronym_df.index.isin(candidate_index)]

	return results


def run():
	cursor = pg.return_postgres_cursor()

	candidate_df = pull_all_acronyms(cursor)

	# full_snomed_query = "select conceptid, term from annotation.active_concept_descriptions"

	# snomed_df = pg.return_df_from_query(cursor, full_snomed_query, (), ['conceptid', 'term'])
	# print(snomed_df[snomed_df['term'] == candidates_all_df.loc[0]['acronym']])
	print(len(candidate_df))
	# print(u.pprint(candidates_all_df))
	# filtered_candidates = de_dupe_and_filter(candidates_all_df, snomed_df)

	# print snomed_df['term']
	u.pprint(candidate_df)
	# print(len(filtered_candidates))
	# for index,row in candidates_all_df.iterrows():
	# 	if type(row['acronym']) != list:
	# 		u.pprint(row)

	# 		try:
	# 			row['acronym'][0]

	# 		except:
	# 			u.pprint(row)

if __name__ == "__main__":
	t = u.Timer("full")
	run()
	t.stop()