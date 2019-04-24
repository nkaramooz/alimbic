set schema 'annotation';
drop table if exists treatment_recs_final;

create table treatment_recs_final as (

	select condition_id, treatment_id, avg(score) as score
	from (
		select condition_id, treatment_id, avg(score) as score
		from annotation.raw_treatment_recs
		group by condition_id, treatment_id, pmid
	) tb1
	group by condition_id, treatment_id
	having avg(score) >= 0.50
);

create index tx_recs_final_cid on treatment_recs_final(condition_id);
create index tx_recs_final_tid on treatment_recs_final(treatment_id);
