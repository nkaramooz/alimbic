set schema 'spacy';
drop table if exists entity_linking_train;

create table entity_linking_train as (
   select
   	t1.sentence_id
   	,t1.train as linking_train
      ,t2.train as entity_train
   	,t1.rand
   	,t1.ver
   from spacy.entity_linking_all t1
   join spacy.concept_ner_all t2 
   on t1.sentence_id=t2.sentence_id
   where t1.rand <= 0.99
);

create index entity_linking_train_sentence_id_ind on entity_linking_train(sentence_id);
create index entity_linking_train_ver_ind on entity_linking_train(ver);

