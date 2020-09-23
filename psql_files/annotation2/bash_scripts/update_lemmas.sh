#!/bin/bash
echo "here we go"

echo "insert_new_concepts_to_root.sql"
psql -d laso -f ../insert_to_root/insert_new_concepts_to_upstream_root.sql -q

echo "insert_snomed_did_to_root.sql"
psql -d laso -f ../insert_to_root/insert_snomed_did_to_upstream_root.sql -q
echo "insert_therapies_syn_to_root.sql"
psql -d laso -f ../insert_to_root/insert_therapies_syn_to_upstream_root.sql -q
echo "insert_acronyms_to_root.sql"
psql -d laso -f ../insert_to_root/insert_acronyms_to_upstream_root.sql -q
echo "insert_root_new_desc_to_root.sql"
psql -d laso -f ../insert_to_root/insert_root_new_desc_to_root.sql -q
echo "insert_to_downstream.sql"
psql -d laso -f ../insert_to_root/insert_to_downstream_root.sql -q
echo "add_adid_acronym.sql"
psql -d laso -f ../manual_tables/add_adid_acronym.sql -q
echo "drop lemmas"
psql -d laso -c "drop table annotation2.lemmas"
echo "update_lemmas.py"
python /Users/LilNimster/Documents/cascade/psql_files/annotation2/update_lemmas.py