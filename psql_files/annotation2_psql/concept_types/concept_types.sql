set schema 'annotation2';

drop table if exists concept_types;

create table concept_types (
  	root_acid varchar(40)
	,rel_type text
	,active integer
	,effectivetime timestamp
);

insert into concept_types

	select
		root_acid
		,rel_type
		,active
		,effectivetime
	from (

		select
			root_acid
			,rel_type
			,active
			,effectivetime
			,row_number () over (partition by root_acid, rel_type order by effectivetime desc) as row_num
		from (
			select 
				root_acid
				,rel_type
				,active
				,effectivetime
			from annotation2.base_concept_types

			union

			select
				root_acid
				,rel_type
				,active
				,effectivetime
			from annotation2.concept_types_app
			) t1
		) t2
	where row_num = 1
;

create index if not exists concept_types_root_acid_ind on annotation2.concept_types(root_acid);
create index if not exists concept_types_rel_type_ind on annotation2.concept_types(rel_type);
create index if not exists concept_types_active_ind on annotation2.concept_types(active);