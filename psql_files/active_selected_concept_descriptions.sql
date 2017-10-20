set schema 'annotation';

drop table if exists active_selected_concept_descriptions;
create table active_selected_concept_descriptions as (
	select 
		terms.id
		,tag.conceptid
		,terms.term
	from annotation.active_selected_concepts tag
	join annotation.active_descriptions terms
		on tag.conceptid = terms.conceptid
);