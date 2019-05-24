set schema 'annotation';
drop table if exists title_treatment_candidates_filtered;

create table title_treatment_candidates_filtered as (
	select *
	from annotation.title_treatment_candidates t1
	where not exists (select 1 from snomed.curr_transitive_closure_f
		where subtypeid != '257556004' and subtypeid = t1.treatment_id and supertypeid in ('362981000', '123037004', '404684003', '363787002', '48176007'))
);

create index title_f_condition_id on title_treatment_candidates_filtered(condition_id);
create index title_f_pmid on title_treatment_candidates_filtered(pmid);
create index title_f_id on title_treatment_candidates_filtered(id);
create index title_f_treatment_id on title_treatment_candidates_filtered(treatment_id);