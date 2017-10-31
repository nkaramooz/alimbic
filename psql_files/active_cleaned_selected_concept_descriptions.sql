set schema 'annotation';
drop table if exists active_cleaned_selected_concept_descriptions;

create table active_cleaned_selected_concept_descriptions as (
	select 
		distinct on (conceptid, term) 
		id
		,conceptid
		,term 
	from annotation.active_cleaned_selected_concept_descriptions_prelim
);