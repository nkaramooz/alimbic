-- Create index for new versions of the pubmedx corpus. This should only be run once after baseline files are indexed

set schema 'pubmed';

create index if not exists sentence_tuples_2_sentence_id_ind on sentence_tuples_2(sentence_id);
create index if not exists sentence_tuples_2_section_ind on sentence_tuples_2(section);
create index if not exists sentence_tuples_2_section_ind_ind on sentence_tuples_2(section_ind);
create index if not exists sentence_tuples_2_pmid_ind on sentence_tuples_2(pmid);
create index if not exists sentence_tuples_2_journal_pub_year_ind on sentence_tuples_2(journal_pub_year);
create index if not exists sentence_tuples_2_journal_iso_abbrev_ind on sentence_tuples_2(journal_iso_abbrev);
create index if not exists sentence_tuples_2_ver_ind on sentence_tuples_2(ver);


create index if not exists sentence_annotations_2_sentence_id_ind on sentence_annotations_2(sentence_id);
create index if not exists sentence_annotations_2_section_ind on sentence_annotations_2(section);
create index if not exists sentence_annotations_2_section_ind_ind on sentence_annotations_2(section_ind);
create index if not exists sentence_annotations_2_acid_ind on sentence_annotations_2(acid);
create index if not exists sentence_annotations_2_adid_ind on sentence_annotations_2(adid);
create index if not exists sentence_annotations_2_final_ann_ind on sentence_annotations_2(final_ann);
create index if not exists sentence_annotations_2_pmid_ind on sentence_annotations_2(pmid);
create index if not exists sentence_annotations_2_journal_pub_year_ind on sentence_annotations_2(journal_pub_year);
create index if not exists sentence_annotations_2_journal_iso_abbrev_ind on sentence_annotations_2(journal_iso_abbrev);
create index if not exists sentence_annotations_2_ver_ind on sentence_annotations_2(ver);


create index if not exists sentence_concept_arr_2_sentence_id_ind on sentence_concept_arr_2(sentence_id);
create index if not exists sentence_concept_arr_2_section_ind on sentence_concept_arr_2(section);
create index if not exists sentence_concept_arr_2_section_ind_ind on sentence_concept_arr_2(section_ind);
create index if not exists sentence_concept_arr_2_pmid_ind on sentence_concept_arr_2(pmid);
create index if not exists sentence_concept_arr_2_journal_pub_year_ind on sentence_concept_arr_2(journal_pub_year);
create index if not exists sentence_concept_arr_2_journal_iso_abbrev_ind on sentence_concept_arr_2(journal_iso_abbrev);
create index if not exists sentence_concept_arr_2_ver_ind on sentence_concept_arr_2(ver);