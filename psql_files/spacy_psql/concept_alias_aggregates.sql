set schema 'spacy';
drop table if exists concept_alias_aggregates;
create table concept_alias_aggregates (
	term varchar(400) not null
	,a_cid_aggs text[] 
);

insert into concept_alias_aggregates
	select
		distinct(term)
		,array_agg(a_cid)
	from spacy.concept_aliases
	group by term
;