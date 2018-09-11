set schema 'annotation';
drop table if exists sentences;
create table sentences(

  id uuid 
  ,conceptid varchar(40)
  ,concept_arr varchar(40)[]
  ,section text
  ,line_num integer
  ,sentence text
  ,sentence_tuples text[]
);

create index s_id on sentences(id);
create index s_conceptid on sentences(conceptid);
create index s_section on sentences(section);