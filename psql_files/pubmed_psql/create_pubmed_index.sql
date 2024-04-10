-- Create index for new versions of the pubmedx corpus. This should only be run once after baseline files are indexed

set schema 'pubmed';

create index if not exists sentence_tuples_sentence_id_ind on sentence_tuples(sentence_id);
create index if not exists sentence_tuples_section_ind on sentence_tuples(section);
create index if not exists sentence_tuples_section_ind_ind on sentence_tuples(section_ind);
create index if not exists sentence_tuples_pmid_ind on sentence_tuples(pmid);
create index if not exists sentence_tuples_journal_pub_month_ind on sentence_tuples(journal_pub_month);
create index if not exists sentence_tuples_journal_pub_year_ind on sentence_tuples(journal_pub_year);
create index if not exists sentence_tuples_journal_iso_abbrev_ind on sentence_tuples(journal_iso_abbrev);
create index if not exists sentence_tuples_ver_ind on sentence_tuples(ver);


create index if not exists sentence_annotations_sentence_id_ind on sentence_annotations(sentence_id);
create index if not exists sentence_annotations_section_ind on sentence_annotations(section);
create index if not exists sentence_annotations_section_ind_ind on sentence_annotations(section_ind);
create index if not exists sentence_annotations_acid_ind on sentence_annotations(acid);
create index if not exists sentence_annotations_adid_ind on sentence_annotations(adid);
create index if not exists sentence_annotations_final_ann_ind on sentence_annotations(final_ann);
create index if not exists sentence_annotations_pmid_ind on sentence_annotations(pmid);
create index if not exists sentence_annotations_journal_pub_month_ind on sentence_annotations(journal_pub_month);
create index if not exists sentence_annotations_journal_pub_year_ind on sentence_annotations(journal_pub_year);
create index if not exists sentence_annotations_journal_iso_abbrev_ind on sentence_annotations(journal_iso_abbrev);
create index if not exists sentence_annotations_ver_ind on sentence_annotations(ver);


create index if not exists sentence_concept_arr_sentence_id_ind on sentence_concept_arr(sentence_id);
create index if not exists sentence_concept_arr_section_ind on sentence_concept_arr(section);
create index if not exists sentence_concept_arr_section_ind_ind on sentence_concept_arr(section_ind);
create index if not exists sentence_concept_arr_concept_arr_ind on sentence_concept_arr(concept_arr);
create index if not exists sentence_concept_arr_pmid_ind on sentence_concept_arr(pmid);
create index if not exists sentence_concept_arr_journal_pub_month_ind on sentence_concept_arr(journal_pub_month);
create index if not exists sentence_concept_arr_journal_pub_year_ind on sentence_concept_arr(journal_pub_year);
create index if not exists sentence_concept_arr_journal_iso_abbrev_ind on sentence_concept_arr(journal_iso_abbrev);
create index if not exists sentence_concept_arr_ver_ind on sentence_concept_arr(ver);


create index if not exists abstract_tuples_pmid_ind on abstract_tuples(pmid);
create index if not exists abstract_tuples_journal_pub_month_ind on abstract_tuples(journal_pub_month);
create index if not exists abstract_tuples_journal_pub_year_ind on abstract_tuples(journal_pub_year);
create index if not exists abstract_tuples_journal_iso_abbrev_ind on abstract_tuples(journal_iso_abbrev);


create index if not exists abstract_concept_arr_pmid_ind on abstract_concept_arr(pmid);
create index if not exists abstract_concept_arr_journal_pub_month_ind on abstract_concept_arr(journal_pub_month);
create index if not exists abstract_concept_arr_journal_pub_year_ind on abstract_concept_arr(journal_pub_year);
create index if not exists abstract_concept_arr_journal_iso_abbrev_ind on abstract_concept_arr(journal_iso_abbrev);
create index if not exists abstract_concept_arr_abs_concept_arr_ind on abstract_concept_arr(abs_concept_arr);	