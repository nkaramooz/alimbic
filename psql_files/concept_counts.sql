set schema 'annotation';

drop table if exists concept_counts;
create table concept_counts as (
    select conceptid, count(*) as count
    from annotation.sentences3
    group by conceptid
);