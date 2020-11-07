#!/bin/bash

echo "full_relationship.sql"
psql -d alimbic -f full_relationship.sql -q
echo "export full_relationship"
psql -d alimbic -c "\copy snomed2.full_relationship to '/home/nkaramooz/Documents/alimbic/psql_files/snomed2_psql/snomed_manual_tables/full_relationship.txt' DELIMITER E'\t'"
echo "transitive_closure script"
perl new_transitive_closure.pl full_relationship.txt transitive_closure.txt
echo "transitive_closure_table_declarations.sql"
psql -d alimbic -f "transitive_closure_table_declarations.sql"
echo "load transitive_closure"
psql -d alimbic -c "COPY snomed2.transitive_closure_cid FROM '/home/nkaramooz/Documents/alimbic/psql_files/snomed2_psql/snomed_manual_tables/transitive_closure.txt' DELIMITER E'\t'"
echo "transitive_closure_acid.sql"
psql -d alimbic -f "transitive_closure_acid.sql"
echo "full_relationship_acid.sql"
psql -d alimbic -f "full_relationship_acid.sql"