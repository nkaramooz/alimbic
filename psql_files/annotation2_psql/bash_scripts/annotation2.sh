#!/bin/bash
echo "here we go"
psql -d alimbic -c "drop table annotation2.cleaned_selected_descriptions_prelim;"
echo "running manual_description_filters.py"
python /home/nkaramooz/Documents/alimbic/manual_description_filters.py

echo "cleaned_selected_descriptions_de_duped.sql"
psql -d alimbic -f ../cleaned_selected_descriptions_de_duped.sql -q
echo "snomed_synonyms.sql"
psql -d alimbic -f ../snomed_synonyms.sql -q
echo "snomed_cid_ignore.sql"
psql -d alimbic -f ../snomed_cid_ignore.sql -q
echo "acronym_augmented_descriptions.sql"
psql -d alimbic -f ../acronym_augmented_descriptions.sql -q

echo "insert_snomed_cid_to_root.sql"
psql -d alimbic -f ../insert_to_root/insert_snomed_cid_to_upstream_root.sql -q
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
echo "insert_to_downstream.sql"
psql -d alimbic -f ../insert_to_root/insert_to_downstream_root.sql -q
echo "add_adid_acronym.sql"
psql -d alimbic -f ../manual_tables/add_adid_acronym.sql -q
echo "declare custom terms"
psql -d alimbic -f ../manual_tables/declare_custom_terms.sql -q
echo "create_custom_terms.py"
python /home/nkaramooz/Documents/alimbic/psql_files/annotation2_psql/manual_tables/create_custom_terms.py
echo "rerun add_adid_acronym.sql now with custom_terms"
psql -d alimbic -f ../manual_tables/add_adid_acronym.sql -q
echo "lemmatizer.py"
python /home/nkaramooz/Documents/alimbic/psql_files/annotation2_psql/bash_scripts/lemmatizer.py
echo "first_word"
psql -d alimbic -f ../annotator_tables/first_word.sql