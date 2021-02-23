set schema 'ml2';
drop table if exists treatment_recs_final_2;

create table treatment_recs_final_2 as (

	select 
		condition_acid
		,treatment_acid
		,avg(score)
	from ml2.treatment_recs_staging
	group by condition_acid, treatment_acid
	having avg(score) >= 0.5
);

create index tx_recs_final_2_cid on treatment_recs_final_2(condition_acid);
create index tx_recs_final_2_tid on treatment_recs_final_2(treatment_acid);
