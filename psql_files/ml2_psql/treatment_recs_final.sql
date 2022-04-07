set schema 'ml2';
drop table if exists treatment_recs_final_1;

create table treatment_recs_final_1 as (

	select 
		condition_acid
		,treatment_acid
		,avg(score)
	from ml2.treatment_recs_staging_2
	group by condition_acid, treatment_acid
	having avg(score) > 0.5 and count(*) >= 2
);

create index tx_recs_final_1_cid on treatment_recs_final_1(condition_acid);
create index tx_recs_final_1_tid on treatment_recs_final_1(treatment_acid);
