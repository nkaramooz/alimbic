set schema 'annotation';

drop table if exists distinct_sentences;
create table distinct_sentences as (
   select 
   	public.uuid_generate_v4() as id
   	,sentence_tuples
   	,0 as version
   from (select id, sentence_tuples, row_number() over (partition by id) as row_num from annotation.sentences5) tb1 
   where tb1.row_num=1
);

create index distinct_sentences_version_ind on distinct_sentences(version);
create index distinct_sentences_id_ind on distinct_sentences(id);


