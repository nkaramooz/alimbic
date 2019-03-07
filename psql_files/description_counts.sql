set schema 'annotation';

drop table if exists description_counts;
create table description_counts (
   did varchar(40)
  ,count integer
);