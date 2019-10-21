set schema 'annotation';

drop table if exists labelled_treatments;
create table labelled_treatments as (
	select
	public.uuid_generate_v4() as id
	,condition_id
	,treatment_id
	,label
	,0 as ver
	from (
	select condition_id, treatment_id, label
	from annotation.labelled_treatments_seed

	union

	select condition_id, treatment_id, label
	from annotation.labelled_treatments_app
	) t1
);
