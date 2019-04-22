set schema 'annotation';

drop table if exists distinct_sentences;
create table distinct_sentences as (
   select sentence_tuples 
   from (select id, sentence_tuples, row_number() over (partition by id) as row_num from annotation.sentences4) tb1 
   where tb1.row_num=1
);