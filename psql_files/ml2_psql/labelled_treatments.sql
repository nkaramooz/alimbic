set schema 'ml2';

-- drop table if exists labelled_treatments;

-- create table labelled_treatments (
-- 	id varchar(40)
--   ,condition_acid varchar(40)
--   ,treatment_acid varchar(40)
--   ,label integer
--   ,ver integer
-- );


insert into labelled_treatments 
	select
		public.uuid_generate_v4() as id
		,t1.condition_acid
		,t1.treatment_acid
		,t1.label
		,0 as ver
		from ml2.labelled_treatments_seed t1
		left join ml2.labelled_treatments t2
		on t1.condition_acid = t2.condition_acid 
			and t1.treatment_acid = t2.treatment_acid
			and t1.label = t2.label
		where t2.label is null
;

insert into labelled_treatments
	select
		public.uuid_generate_v4() as id
		,t1.condition_acid
		,t1.treatment_acid
		,t1.label
		,0 as ver
		from ml2.labelled_treatments_app t1
		left outer join ml2.labelled_treatments t2
		on t1.condition_acid = t2.condition_acid 
			and t1.treatment_acid = t2.treatment_acid
			and t1.label = t2.label
	where t2.label is null
;
