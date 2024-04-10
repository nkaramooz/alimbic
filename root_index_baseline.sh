python3 pubmed_processor.py

psql -d alimbic -f psql_files/pubmed_psql/create_pubmed_index.sql -q

psql -d alimbic -f psql_files/annotation2_psql/annotator_tables/concept_counts.sql -q

psql -d alimbic -f psql_files/annotation2_psql/annotator_tables/description_counts.sql -q

psql -d alimbic -f psql_files/annotation2_psql/annotator_tables/preferred_concept_names.sql -q

psql -d alimbic -f psql_files/ml2_psql/word_counts_50k.sql -q

python3 ml2.py