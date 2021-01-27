set schema 'annotation2';

drop table if exists concept_counts;
create table concept_counts(
	concept varchar(400) not null
	,cnt integer not null
	,unique(concept)
);
create index concept_counts_concept_ind on concept_counts(concept);


insert into concept_counts
	select
		final_ann as concept
		,count(*) as cnt
	from pubmed.sentence_annotations_1_8
	group by final_ann
;