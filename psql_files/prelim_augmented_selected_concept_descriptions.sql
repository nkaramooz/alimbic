set schema 'annotation';

drop table if exists prelim_augmented_selected_concept_descriptions;
create table prelim_augmented_selected_concept_descriptions as (

	select 
		id
		,conceptid
		,term
		,'1'::text as active
		,null::timestamp as effectivetime
	from annotation.active_cleaned_selected_concept_descriptions

	union

	select
		id::text
		,conceptid
		,term
		,'1'::text as active
		,null::timestamp as effectivetime
	from annotation.filtered_augmented_descriptions

	union

	select 
		description_id as id
		,conceptid
		,term
		,'1'::text as active
		,null::timestamp as effectivetime
	from annotation.description_whitelist

	union 

	select 
		description_id as id
		,conceptid 
		,description as term 
		,'1'::text as active
		,effectivetime
	from annotation.new_concepts

	union

	select 
		description_id as id
		,conceptid 
		,term 
		,'0'::text as active
		,null::timestamp as effectivetime 			
	from annotation.description_blacklist
);

create index if not exists ind_prelim_aug_description_id on prelim_augmented_selected_concept_descriptions(id);
create index if not exists ind_prelim_aug_effectivetime on prelim_augmented_selected_concept_descriptions(effectivetime);
create index if not exists ind_prelim_aug_active on prelim_augmented_selected_concept_descriptions(active);