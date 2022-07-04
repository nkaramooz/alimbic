set schema 'spacy';
drop table if exists concept_ner_validation;
create table concept_ner_validation as (
   select
   	sentence_id
   	,train
   	,rand
   	,ver
   from spacy.concept_ner_all
   where rand > 0.99 and rand < 0.991
);

create index concept_ner_validation_sentence_id_ind on concept_ner_validation(sentence_id);

