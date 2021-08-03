#!/bin/bash

rm ./ftp.ncbi.nlm.nih.gov -r
wget -r ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline
gunzip ./ftp.ncbi.nlm.nih.gov/pubmed/baseline/*.gz || exit 1
rm ./ftp.ncbi.nlm.nih.gov/pubmed/baseline/*.md5 || exit 1
rm ./ftp.ncbi.nlm.nih.gov/pubmed/baseline/README.txt || exit 1
rm ~/Documents/alimbic/resources/pubmed_baseline/ftp.ncbi.nlm.gov -r
mv ftp.ncbi.nlm.nih.gov/pubmed/baseline ~/Documents/alimbic/resources/pubmed_baseline/ftp.ncbi.nlm.nih.gov