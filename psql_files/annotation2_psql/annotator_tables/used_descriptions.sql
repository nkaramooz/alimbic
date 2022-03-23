set schema 'annotation2';

drop table if exists used_descriptions;
create table used_descriptions as (
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
			where tb2.term != upper(tb2.term)
			) tb3
		) tb4
	join annotation2.downstream_root_did tb5
	on tb5.adid = tb4.adid
);

	create index if not exists used_descriptions_cid on used_descriptions(acid);
		create index if not exists used_descriptions_term on used_descriptions(term);