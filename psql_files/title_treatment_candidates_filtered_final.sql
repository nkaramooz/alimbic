set schema 'annotation';
drop table if exists title_treatment_candidates_filtered_final;

create table title_treatment_candidates_filtered_final as (
	select distinct on (condition_id, treatment_id, sentence) *
	from annotation.title_treatment_candidates t1

);

create index title_f_condition_id_f on title_treatment_candidates_filtered_final(condition_id);
create index title_f_pmid_f on title_treatment_candidates_filtered_final(pmid);
create index title_f_id_f on title_treatment_candidates_filtered_final(id);
create index title_f_treatment_id_f on title_treatment_candidates_filtered_final(treatment_id);