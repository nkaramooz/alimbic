set schema 'ml2';
drop table if exists test_sentences;


create table test_sentences as (
    select 
    	id
    	,sentence_id
    	,section_ind
    	,x_train_gen
    	,label
    	,ver_gen
    from ml2.all_training_sentences
    where rand > 0.97
);

create index if not exists test_sentences_label_ind on test_sentences(label);
create index if not exists test_sentences_ver_gen_ind on test_sentences(ver_gen);
create index if not exists test_sentences_id_ind on test_sentences(id);
