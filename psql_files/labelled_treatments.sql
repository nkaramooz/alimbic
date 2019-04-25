set schema 'annotation';

drop table if exists labelled_treatments;
create table labelled_treatments as (
	select condition_id, treatment_id, label
	from annotation.labelled_treatments_seed

	union

	select condition_id, treatment_id, label
	from annotation.labelled_treatments_app
);
