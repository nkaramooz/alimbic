set schema 'spacy';
drop table if exists wiki_spacy_entities;

create table wiki_spacy_entities as (
	select
		a_cid
		,term
		,description
	from (
		select row_number () over (partition by a_cid) as row_num
		,a_cid
		,term
		,t1.desc as description
		from spacy.entities t1
		) t2
	where row_num = 1
);