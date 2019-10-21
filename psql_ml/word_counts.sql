set schema 'annotation';

drop table if exists word_counts;
create table word_counts( 
   id varchar(40) UNIQUE
   ,word varchar(40) UNIQUE
   ,count int DEFAULT 0
);


