#!/bin/bash
echo "here we go"
psql -d laso -c "drop table annotation2.cleaned_selected_descriptions_prelim;"
echo "running manual_description_filters.py"
python /Users/LilNimster/Documents/cascade/manual_description_filters.py

echo "cleaned_selected_descriptions_de_duped.sql"
psql -d laso -f ../cleaned_selected_descriptions_de_duped.sql -q
echo "snomed_synonyms.sql"
psql -d laso -f ../snomed_synonyms.sql -q
echo "snomed_cid_ignore.sql"
psql -d laso -f ../snomed_cid_ignore.sql -q
echo "acronym_augmented_descriptions.sql"
psql -d laso -f ../acronym_augmented_descriptions.sql -q

echo "insert_snomed_cid_to_root.sql"
psql -d laso -f ../insert_to_root/insert_snomed_cid_to_root.sql -q
echo "insert_new_concepts_to_root.sql"
psql -d laso -f ../insert_to_root/insert_new_concepts_to_root.sql -q

echo "insert_snomed_did_to_root.sql"
psql -d laso -f ../insert_to_root/insert_snomed_did_to_root.sql -q
echo "insert_therapies_syn_to_root.sql"
psql -d laso -f ../insert_to_root/insert_therapies_syn_to_root.sql -q
echo "insert_acronyms_to_root.sql"
psql -d laso -f ../insert_to_root/insert_acronyms_to_root.sql -q
echo "insert_manual_active_desc_to_root.sql"
psql -d laso -f ../insert_to_root/insert_manual_active_desc_to_root.sql -q