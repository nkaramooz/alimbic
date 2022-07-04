set schema 'spacy';
drop table if exists concept_ner_tuples;

create table if not exists concept_ner_tuples (
	 sentence_id varchar(40) not null
	,sentence_tuples jsonb
	,rand float
	,ver integer
);

insert into concept_ner_tuples
	select
		sentence_id
		,sentence_tuples
		,random() as rand
		,0 as ver
	from pubmed.sentence_tuples_2
;

create index concept_ner_tuples_ver_ind on spacy.concept_ner_tuples(ver);
create index concept_ner_tuples_sentence_id_ind on spacy.concept_ner_tuples(sentence_id);