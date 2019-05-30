set schema 'annotation';
drop table if exists treatment_recs_staging;

create table treatment_recs_staging as (

	select condition_id, treatment_id, avg(score) as score
	from (
		select condition_id, treatment_id, avg(score) as score
		from annotation.raw_treatment_recs_staging_2
		group by condition_id, treatment_id, pmid
	) tb1
	group by condition_id, treatment_id
	having avg(score) >= 0.50
);

create index tx_recs_staging_cid on treatment_recs_staging(condition_id);
create index tx_recs_staging_tid on treatment_recs_staging(treatment_id);
