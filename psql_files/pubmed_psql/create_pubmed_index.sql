-- Create index for new versions of the pubmedx corpus. This should only be run once after baseline files are indexed

set schema 'pubmed';

create index if not exists sentence_tuples_1_9_sentence_id_ind on sentence_tuples_1_9(sentence_id);
create index if not exists sentence_tuples_1_9_section_ind on sentence_tuples_1_9(section);
create index if not exists sentence_tuples_1_9_section_ind_ind on sentence_tuples_1_9(section_ind);
create index if not exists sentence_tuples_1_9_pmid_ind on sentence_tuples_1_9(pmid);
create index if not exists sentence_tuples_1_9_journal_pub_year_ind on sentence_tuples_1_9(journal_pub_year);
create index if not exists sentence_tuples_1_9_journal_iso_abbrev_ind on sentence_tuples_1_9(journal_iso_abbrev);
create index if not exists sentence_tuples_1_9_ver_ind on sentence_tuples_1_9(ver);


create index if not exists sentence_annotations_1_9_sentence_id_ind on sentence_annotations_1_9(sentence_id);
create index if not exists sentence_annotations_1_9_section_ind on sentence_annotations_1_9(section);
create index if not exists sentence_annotations_1_9_section_ind_ind on sentence_annotations_1_9(section_ind);
create index if not exists sentence_annotations_1_9_acid_ind on sentence_annotations_1_9(acid);
create index if not exists sentence_annotations_1_9_adid_ind on sentence_annotations_1_9(adid);
create index if not exists sentence_annotations_1_9_final_ann_ind on sentence_annotations_1_9(final_ann);
create index if not exists sentence_annotations_1_9_pmid_ind on sentence_annotations_1_9(pmid);
create index if not exists sentence_annotations_1_9_journal_pub_year_ind on sentence_annotations_1_9(journal_pub_year);
create index if not exists sentence_annotations_1_9_journal_iso_abbrev_ind on sentence_annotations_1_9(journal_iso_abbrev);
create index if not exists sentence_annotations_1_9_ver_ind on sentence_annotations_1_9(ver);


create index if not exists sentence_concept_arr_1_9_sentence_id_ind on sentence_concept_arr_1_9(sentence_id);
create index if not exists sentence_concept_arr_1_9_section_ind on sentence_concept_arr_1_9(section);
create index if not exists sentence_concept_arr_1_9_section_ind_ind on sentence_concept_arr_1_9(section_ind);
create index if not exists sentence_concept_arr_1_9_pmid_ind on sentence_concept_arr_1_9(pmid);
create index if not exists sentence_concept_arr_1_9_journal_pub_year_ind on sentence_concept_arr_1_9(journal_pub_year);
create index if not exists sentence_concept_arr_1_9_journal_iso_abbrev_ind on sentence_concept_arr_1_9(journal_iso_abbrev);
create index if not exists sentence_concept_arr_1_9_ver_ind on sentence_concept_arr_1_9(ver);