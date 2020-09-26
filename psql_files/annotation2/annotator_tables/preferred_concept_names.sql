set schema 'annotation2';

drop table if exists preferred_concept_names;
create table preferred_concept_names as (
	select
	tb3.acid
	,tb4.term
	from (
		select
		acid
		,adid
		from (
			select
			acid
			,adid
			,row_number() over (partition by acid, adid order by cnt desc) as rn_num
			from (
				select
				acid
				,adid
				,count(*) as cnt
				from pubmed.sentence_annotations
				group by acid, adid
				) tb1
			) tb2
		where rn_num = 1
		) tb3
	join annotation2.downstream_root_did tb4
	on tb3.adid = tb4.adid
);

	create index if not exists concept_names_cid on preferred_concept_names(acid);
		create index if not exists concept_names on preferred_concept_names(term);