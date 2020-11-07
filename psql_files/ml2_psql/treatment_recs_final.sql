set schema 'ml2';
drop table if exists treatment_recs_final;

create table treatment_recs_final as (

	select 
		condition_acid
		,treatment_acid
		,avg(score)
	from ml2.treatment_recs_staging
	group by condition_acid, treatment_acid
	having avg(score) > 0.6 and count(*) > 1
);

create index tx_recs_final_cid on treatment_recs_final(condition_acid);
create index tx_recs_final_tid on treatment_recs_final(treatment_acid);
