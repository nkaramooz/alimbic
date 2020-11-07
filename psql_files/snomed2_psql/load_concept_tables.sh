#!/bin/bash

echo "declare tables"
psql -d alimbic -f sn2_table_declarations.sql

echo "load active concepts"
psql -d alimbic -f selected_concepts.sql
psql -d alimbic -f active_selected_concepts.sql

echo "load active descriptions"
psql -d alimbic -f active_descriptions.sql
psql -d alimbic -f active_selected_descriptions.sql