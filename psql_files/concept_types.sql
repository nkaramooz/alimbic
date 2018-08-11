set schema 'annotation';

drop table if exists concept_types;

create table concept_types as(
	select 
		root_cid
		,associated_cid
		,rel_type
		,active
		,effectivetime
	from annotation.base_concept_types

	union all

	select
		root_cid
		,associated_cid
		,rel_type
		,active
		,effectivetime
	from annotation.override_concept_types

);


create index ct_conceptid on concept_types(root_cid);
create index ct_concept_type on concept_types(rel_type)
