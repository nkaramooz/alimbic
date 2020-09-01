#!/bin/bash

echo "declare tables"
psql -d laso -f sn2_table_declarations.sql

echo "load active concepts"
psql -d laso -f selected_concepts.sql
psql -d laso -f active_selected_concepts.sql

echo "load active descriptions"
psql -d laso -f active_descriptions.sql
psql -d laso -f active_selected_descriptions.sql