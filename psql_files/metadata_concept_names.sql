set schema 'snomed';

drop table if exists metadata_concept_names;
create table metadata_concept_names as (
	select 
		tag.conceptid
		,terms.term
	from snomed.tagging_concepts tag
	join snomed.active_concept_names terms
		on tag.conceptid = terms.conceptid
);