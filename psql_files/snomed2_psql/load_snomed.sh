#!/bin/bash
echo "Hello, world!"
psql -d alimbic -f environment-postgresql.sql
psql -d alimbic -f load-postgresql.sql
# psql -d laso -f load_transitive_closure.sql
# psql -d laso -f active_snomed_relationships.sql



