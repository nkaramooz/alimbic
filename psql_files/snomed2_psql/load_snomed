#!/bin/bash
echo "Hello, world!"
psql -d laso -f environment-postgresql.sql
psql -d laso -f load-postgresql.sql
psql -d laso -f load_transitive_closure.sql



