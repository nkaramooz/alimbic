set schema 'spacy';

create table concept_ner_train as (
   select
   	sentence_id
   	,train
   	,rand
   	,ver
   from spacy.concept_ner_all
   where rand <= 0.99
);

create index concept_ner_train_sentence_id_ind on concept_ner_train(sentence_id);
create index concept_ner_train_ver_ind on concept_ner_train(ver);

