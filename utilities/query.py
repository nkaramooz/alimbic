class Query:
	def __init__(self, **kwargs):
		# Form [[[]], [[]]]
		self.full_query_concepts_list = kwargs.get('full_query_concepts_list', None)
		self.flattened_concept_list = kwargs.get('flattened_concept_list', None)
		# Form [[],[]]
		self.flattened_query = kwargs.get('flattened_query', None)
		self.unmatched_list = kwargs.get('unmatched_list', None)
		self.filters = kwargs.get('filters', [])
		self.query_types_list = kwargs.get('query_types_list', None)
		self.root_query = kwargs.get('root_query', None)
		self.pivot_history = kwargs.get('pivot_history', [])

#query_type = {'condition', 'treatment', 'organism', 'diagnostic'}
def get_pivot_queries(query_type, a_cid_candidates, query_concept_list, pivot_list, cursor):
	a_cid_candidates_string = '(' + ', '.join(f'\'{item}\'' for item in a_cid_candidates) + ')'
	query_concept_string = '(' + ', '.join(f'\'{item}\'' for item in query_concept_list) + ')'
	pivot_string = '(' + ', '.join(f'\'{item}\'' for item in pivot_list) + ')'

	if query_type == 'condition' or query_type == 'diagnostic' or query_type == 'cause':
		if len(pivot_list) == 0:
			query = """
				select
					root_acid as acid
					,rel_type as concept_type
				from annotation2.concept_types
				where active=1 and 
					rel_type='%s' and
					root_acid in %s
			""" % (query_type, a_cid_candidates_string)
			return query
		else:
			query = """
				select
					root_acid as acid
					,rel_type as concept_type
				from annotation2.concept_types t1
				join (select distinct(child_acid) from snomed2.transitive_closure_acid 
					where parent_acid in (select child_acid from snomed2.transitive_closure_acid where parent_acid in %s)) t2
					on t1.root_acid = t2.child_acid
				where t1.active=1 and 
					rel_type='%s' and
					root_acid in %s
			""" % (pivot_string, query_type, a_cid_candidates_string)
			return query

	elif query_type == 'treatment':
		if len(pivot_list) == 0:
			query = """
				select distinct(treatment_acid) as acid
					,'treatment' as concept_type
				from ml2.treatment_recs_final_1 t1
				join (select root_acid from annotation2.concept_types where active=1 and rel_type='treatment') t2
				on t1.treatment_acid = t2.root_acid
				where condition_acid in %s and treatment_acid in %s 
			""" % (query_concept_string, a_cid_candidates_string)
			return query
		else:
			query = """
				select distinct(treatment_acid) as acid
					,'treatment' as concept_type
				from ml2.treatment_recs_final_1 t1
				join (select root_acid from annotation2.concept_types where active=1 and rel_type='treatment' 
					and root_acid in (select child_acid from snomed2.transitive_closure_acid where parent_acid in %s) ) t2
				on t1.treatment_acid = t2.root_acid
				where condition_acid in %s and treatment_acid in %s 
			""" % (pivot_string, query_concept_string, a_cid_candidates_string)
			return query
	