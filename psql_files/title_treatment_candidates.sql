set schema 'annotation';
drop table if exists title_treatment_candidates;

create table title_treatment_candidates as (
	select tb3.conceptid as condition_id, 
		tb1.conceptid as treatment_id, 
		tb1.sentence, 
		tb1.sentence_tuples, 
		tb1.pmid, 
		tb1.id,
		tb1.section
	from annotation.sentences5 tb1
	join (select tb2.id, tb2.conceptid 
			from annotation.sentences5 tb2
			where conceptid='37796009'
			-- inner join annotation.concept_types tb3
			-- on tb2.conceptid = tb3.root_cid and rel_type in ('condition', 'symptom', 'organism')
	) tb3 
	on tb1.id = tb3.id
	and tb1.conceptid != tb3.conceptid
);

create index title_condition_id on title_treatment_candidates(condition_id);
create index title_pmid on title_treatment_candidates(pmid);
create index title_id on title_treatment_candidates(id);
create index title_treatment_id on title_treatment_candidates(treatment_id);