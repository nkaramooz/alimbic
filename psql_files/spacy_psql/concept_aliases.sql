set schema 'spacy';

drop table if exists concept_aliases;
create table concept_aliases (
    a_cid varchar(36) not null
	,term varchar(400) not null
);

insert into concept_aliases
	select
		distinct on (term, acid)
		acid as a_cid 
		,t1.term
	from annotation2.lemmas t1
	where t1.acid in (select root_acid from annotation2.concept_types
		where rel_type in ('condition', 'chemical', 'treatment', 'outcome', 'statistic', 'symptom', 'diagnostic', 'cause', 'study_design')
		and (active=1 or active=3))
;

create index concept_aliases_a_cid_ind on concept_aliases(a_cid);