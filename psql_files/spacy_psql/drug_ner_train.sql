set schema 'spacy';
drop table if exists drug_ner_sentences;

create table if not exists drug_ner_sentences (
	sentence_id varchar(40) not null
	,sentence_tuples json
	,ver integer
);

insert into drug_ner_sentences
	select
		t1.sentence_id
		,sentence_tuples
		,0 as ver
	from pubmed.sentence_tuples_2 t1
	right join (
		select distinct(sentence_id) 
		from pubmed.sentence_annotations_2
		where final_ann in (select child_acid from snomed2.transitive_closure_acid where parent_acid='250597')
	) t2
	on t1.sentence_id=t2.sentence_id
;