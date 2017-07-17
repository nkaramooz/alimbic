set schema 'annotation';

drop table if exists selected_concept_names;
create table selected_concept_names as (
	select 
		terms.id
		,tag.conceptid
		,terms.term
	from annotation.selected_concepts tag
	join annotation.active_concept_descriptions terms
		on tag.conceptid = terms.conceptid
);