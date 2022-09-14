set schema 'spacy';
drop table if exists entity_linking_validation;

create table entity_linking_validation as (
   select
   	sentence_id
   	,train
   	,rand
   	,ver
   from spacy.entity_linking_all
   where rand > 0.99
);

create index entity_linking_validation_sentence_id_ind on entity_linking_validation(sentence_id);
create index entity_linking_validation_ver_ind on entity_linking_validation(ver);
