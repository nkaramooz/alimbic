set schema 'annotation';

drop table if exists preferred_concept_names;
create table preferred_concept_names as (
	select 
		conceptid
		,term
	from (
		select did, conceptid, term, cnt, row_number() over (partition by conceptid order by cnt desc) as rn_num
		from (
			select did, conceptid, term, sum(cnt) as cnt
			from (
				select tb1.did, tb2.conceptid, tb2.term, tb1.count as cnt
				from annotation.description_counts tb1
				left join annotation.lemmas_3 tb2
				on tb1.did = tb2.description_id
			) tb3
		group by did, conceptid, term
		) tb4
	) tb5
	where rn_num = 1

);

create index if not exists concept_names_cid on preferred_concept_names(conceptid);
create index if not exists concept_names on preferred_concept_names(term);