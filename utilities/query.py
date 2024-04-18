class Query:
	def __init__(self, **kwargs):
		self.query_string = kwargs.get('query_string', None)
		
		# Cleaned query string
		self.cleaned_query_string = kwargs.get('cleaned_query_string', None)

		# The 2-d list of explicit concepts annotated in the query. [[],[]]
		self.nested_query_acids = kwargs.get('nested_query_acids', None)
		
		# The explicit concepts in the query in a one-dimensional list. []
		self.flat_query_acids = kwargs.get('flat_query_acids', [])

		# The expanded list of query acids in a 2-d list. [[],[]]
		self.nested_expanded_query_acids = kwargs.get('nested_expanded_query_acids', None)

		# The flattened list of expanded query acids. []
		self.flat_expanded_query_acids = kwargs.get('flat_expanded_query_acids', [])
		
		# Terms in the query that were not matched to a concept
		self.unmatched_terms_list = kwargs.get('unmatched_terms_list', [])

		# Search filters
		self.filters = kwargs.get('filters', [])

		# Concept types in the query
		self.query_concept_types_list = kwargs.get('query_concept_types_list', None)
		
		# The elasticsearch query created from the searched expanded concepts
		# unmatched terms, and filters.
		self.es_query = kwargs.get('es_query', None)
		self.pivot_history = kwargs.get('pivot_history', [])

		# Pivot term is the displayed name of the concept of the pivot item
		# item the user selected. This terms updates the displayed user search query.
		self.pivot_term = kwargs.get('pivot_term', None)
	
	def return_json(self):
		return {
			'query_string' : self.query_string,
			'cleaned_query_string' : self.cleaned_query_string,
			'nested_query_acids' : self.nested_query_acids,
			'flat_query_acids' : self.flat_query_acids,
			'nested_expanded_query_acids' : self.nested_expanded_query_acids,
			'flat_expanded_query_acids' : self.flat_expanded_query_acids,
			'unmatched_terms_list' : self.unmatched_terms_list,
			'query_concept_types_list' : self.query_concept_types_list,
			'pivot_history' : self.pivot_history
		}