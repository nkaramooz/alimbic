set schema 'ml2';
drop table if exists cso;

create table cso as (
	select 
		child_acid as acid
		,case when parent_acid in ('31511', '30892', '123609', '73827', '516552', '47403','72312','452632', '516552', '142136', '214167') then 'organism'
			when parent_acid = '11220' then 'condition'
			when parent_acid = '346270' then 'symptom' end as rel_type
		, 0 as ver
	from snomed2.transitive_closure_acid
	where parent_acid in ('31511', '30892', '123609', '73827', '516552', '47403','72312','452632', '516552', '142136', '214167', '11220', '346270')
);

create index cso_acid_ind on cso(acid);
create index cso_rel_type_ind on cso(rel_type);
create index cso_ver_ind on cso(ver);