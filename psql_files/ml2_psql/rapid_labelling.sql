set schema 'ml2';
drop table if exists ml2.rapid_labelling;
create table rapid_labelling as (

	select 
		t1.condition_acid
		,t3.term as condition 
		,t1.treatment_acid 
		,t4.term as treatment 
		,t1.avg as t1avg
		,t2.avg as t2avg
		,0 as ver
		,t5.cnt 
	from ml2.treatment_recs_final_1 t1
	left join ml2.treatment_recs_final_2 t2 
		on t1.condition_acid=t2.condition_acid and t1.treatment_acid=t2.treatment_acid 
	left join annotation2.preferred_concept_names t3 
		on t1.condition_acid=t3.acid 
	left join annotation2.preferred_concept_names t4 
		on t1.treatment_acid=t4.acid
	left join annotation2.concept_counts t5
		on t1.treatment_acid=t5.concept
	left join annotation2.concept_counts t6
		on t1.condition_acid=t6.concept 
	where t2.condition_acid is null and t1.treatment_acid != '485423' and t5.cnt > 1000 and t6.cnt > 1000
	

);
create index if not exists rapid_labelling_ver on ml2.rapid_labelling(ver);