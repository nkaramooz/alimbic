set schema 'annotation';
drop table if exists tmp;

create table tmp as (
	select tb1.id, tb1.pmid, tb1.conceptid, tb1.concept_arr, tb1.section, tb1.section_index,
		tb1.line_num, tb1.sentence, tb1.sentence_tuples
	from annotation.sentences5 tb1
	join annotation.concept_types tb2
	on tb1.conceptid = tb2.root_cid
	where tb2.rel_type in ('condition', 'symptom', 'organism')
);

create index tmp_id_ind on tmp(id);
create index tmp_conceptid on tmp(conceptid);