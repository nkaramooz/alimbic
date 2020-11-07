#!/bin/bash


csfd=$(python get_update_filenames.py)
for i in $csfd
do
echo ftp://ftp.ncbi.nlm.nih.gov/pubmed/$i
done