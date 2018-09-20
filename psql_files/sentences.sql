set schema 'annotation';
drop table if exists sentences2;
create table sentences2(

  id uuid
  ,pmid varchar(40)
  ,conceptid varchar(40)
  ,concept_arr varchar(40)[]
  ,section text
  ,section_index integer
  ,line_num integer
  ,sentence text
  ,sentence_tuples text[]
);

create index s_id2 on sentences2(id);
create index s_conceptid2 on sentences2(conceptid);
create index s_section2 on sentences2(section);
create index s_pmid2 on sentences2(pmid);
create index s_sec_index2 on sentences2(section_index);