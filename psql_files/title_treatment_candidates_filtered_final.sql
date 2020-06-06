set schema 'annotation';

drop table if exists title_treatment_candidates_filtered_final;

create table title_treatment_candidates_filtered_final as (
	select distinct on (condition_id, treatment_id, sentence) 
		public.uuid_generate_v4()::text as title_treatment_id
		,t1.id as sentence_id
		,t1.condition_id
		,t1.treatment_id
		,t1.sentence_tuples
		,t1.pmid
		,t1.section
		, 0 as ver
	from annotation.title_treatment_candidates_filtered t1
);

create index title_f_id_f on title_treatment_candidates_filtered_final(title_treatment_id);
create index title_f_condition_id_f on title_treatment_candidates_filtered_final(condition_id);
create index title_f_pmid_f on title_treatment_candidates_filtered_final(pmid);
create index title_f_sentence_id_f on title_treatment_candidates_filtered_final(sentence_id);
create index title_f_treatment_id_f on title_treatment_candidates_filtered_final(treatment_id);
create index title_f_ver_f on title_treatment_candidates_filtered_final(ver);