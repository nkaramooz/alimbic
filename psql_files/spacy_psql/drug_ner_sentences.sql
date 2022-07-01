set schema 'spacy';
drop table if exists ner_sentences;

create table if not exists ner_sentences (
	sentence_id varchar(40) not null
	,sentence_tuples json
	,ver integer
);

insert into ner_sentences
	select
		t1.sentence_id
		,sentence_tuples
		,0 as ver
	from pubmed.sentence_tuples_2 t1
	right join (
		select distinct(sentence_id) 
		from pubmed.sentence_annotations_2
		where final_ann in (
			select root_acid 
			from annotation2.concept_types 
			where rel_type='condition'
			or rel_type='symptom'
			or rel_type='outcome'
			or rel_type='treatment'
			or rel_type='statistic')
	) t2
	on t1.sentence_id=t2.sentence_id
;
