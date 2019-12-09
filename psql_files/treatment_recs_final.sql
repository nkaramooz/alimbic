set schema 'annotation';
drop table if exists treatment_recs_final;

create table treatment_recs_final as (

	select root_cid as condition_id, treatment_id, avg(score)
	from (
		select root_cid, subtypeid, treatment_id, score
		from (
			select t1.root_cid, t2.subtypeid 
			from annotation.concept_types t1
			left join snomed.curr_transitive_closure_f t2
				on t1.root_cid = t2.supertypeid
		union 
		select root_cid as root_cid, root_cid as subtypeid
		from annotation.concept_types
		) t3
	left join annotation.raw_treatment_recs_staging_5 t4
		on t3.subtypeid=t4.condition_id
	) t5
group by root_cid, treatment_id
having avg(score) > 0.50
);

create index tx_recs_final_cid on treatment_recs_final(condition_id);
create index tx_recs_final_tid on treatment_recs_final(treatment_id);
