set schema 'annotation2';

drop table if exists preferred_concept_names;
create table preferred_concept_names as (
	select
	tb5.acid
	,tb5.term
	from (
		select
		acid
		,adid
		from (
			select
				tb2.acid
				,tb1.adid
				,row_number() over (partition by acid order by cnt desc) as rn_num
			from annotation2.description_counts tb1
			join annotation2.downstream_root_did tb2
			on tb1.adid = tb2.adid
			) tb3
		where rn_num = 1
		) tb4
	join annotation2.downstream_root_did tb5
	on tb5.adid = tb4.adid
);

	create index if not exists concept_names_cid on preferred_concept_names(acid);
		create index if not exists concept_names on preferred_concept_names(term);