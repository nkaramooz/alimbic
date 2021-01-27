# !/bin/bash

rm ~/Documents/alimbic/resources/pubmed_update/ftp.ncbi.nlm.nih.gov/* -r
csfd=$(python get_update_filenames.py)
for i in $csfd
do
wget ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/$i
done
gunzip ./*.gz || exit 1
mv -f ./*.xml ~/Documents/alimbic/resources/pubmed_update/ftp.ncbi.nlm.nih.gov/


echo "start pubmed_processor"
python3 ../pubmed_processor2.py
echo "finished pubmed_processor"
# psql -d alimbic -f ../psql_files/ml2_psql/treatment_candidates.sql -q
# psql -d alimbic -c "drop table if exists ml2.treatment_recs_staging;"
# python3 ../ml2.py
# psql -d alimbic -f ../psql_files/ml2_psql/treatment_recs_final.sql -q
