set schema 'ml2';

drop table if exists word_counts;
create table word_counts( 
   id varchar(40) UNIQUE
   ,word varchar(40) UNIQUE
   ,cnt int DEFAULT 0
);


