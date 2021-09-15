#!/bin/bash
echo "here we go"

echo "insert_new_concepts_to_root.sql"
psql -d alimbic -f ../insert_to_root/insert_new_concepts_to_upstream_root.sql -q

echo "insert_snomed_did_to_root.sql"
psql -d alimbic -f ../insert_to_root/insert_snomed_did_to_upstream_root.sql -q
echo "insert_therapies_syn_to_root.sql"
psql -d alimbic -f ../insert_to_root/insert_therapies_syn_to_upstream_root.sql -q
echo "insert_acronyms_to_root.sql"
psql -d alimbic -f ../insert_to_root/insert_acronyms_to_upstream_root.sql -q
echo "insert_root_new_desc_to_root.sql"
psql -d alimbic -f ../insert_to_root/insert_root_new_desc_to_root.sql -q

echo "run custom_terms.py"
python ../manual_tables/create_custom_terms.py

echo "insert_custom_terms_to_upstream_root.sql"
psql -d alimbic -f ../insert_to_root/insert_custom_terms_to_upstream_root.sql

echo "insert_to_downstream.sql"
psql -d alimbic -f ../insert_to_root/insert_to_downstream_root.sql -q
echo "add_adid_acronym.sql"
psql -d alimbic -f ../manual_tables/add_adid_acronym.sql -q


echo "drop lemmas"
psql -d alimbic -c "drop table annotation2.lemmas"
echo "update_lemmas.py"
python /home/nkaramooz/Documents/alimbic/psql_files/annotation2_psql/bash_scripts/update_lemmas.py
echo "first_word"
psql -d alimbic -f ../annotator_tables/first_word.sql