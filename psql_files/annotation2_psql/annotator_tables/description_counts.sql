set schema 'annotation';

drop table if exists description_counts;
create table description_counts (
   adid varchar(40)
  ,cnt integer
);

create index description_counts_adid_ind on annotation.description_counts(adid);

insert into annotation.description_counts
	select
		adid
		,count(*) as cnt
	from pubmed.sentence_annotations_1_9
	where adid != '-1'
	group by adid
;

