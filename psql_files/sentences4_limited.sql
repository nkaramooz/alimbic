set schema 'annotation';

drop table if exists sentences4_limited;
create table sentences4_limited as (
   select id, pmid, conceptid, section, section_index, line_num, sentence
   from annotation.sentences4
);

create index s4l_id on sentences4_limited(id);
create index s4l_pmid on sentences4_limited(pmid);
create index s4l_conceptid on sentences4_limited(conceptid);
create index s4l_section on sentences4_limited(section);
create index s4l_section_index on sentences4_limited(section_index);
create index s4l_line_num on sentences4_limited(line_num);