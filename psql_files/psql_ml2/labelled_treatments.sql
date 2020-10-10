set schema 'ml2';

drop table if exists labelled_treatments;
create table labelled_treatments as (
	select
	public.uuid_generate_v4() as id
	,condition_acid
	,treatment_acid
	,label
	,0 as ver
	from (
	select condition_acid, treatment_acid, label
	from ml2.labelled_treatments_seed

	union

	select condition_acid, treatment_acid, label
	from ml2.labelled_treatments_app
	) t1
);