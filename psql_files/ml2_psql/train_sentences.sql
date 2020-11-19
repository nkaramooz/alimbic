set schema 'ml2';
drop table if exists train_sentences;


create table train_sentences as (
    select 
    	id
    	,sentence_id
    	,section_ind
    	,x_train_gen
    	,label
    	,ver_gen
    from ml2.all_training_sentences
    where rand <= 0.97
);

create index if not exists train_sentences_label_ind on train_sentences(label);
create index if not exists train_sentences_ver_gen_ind on train_sentences(ver_gen);
create index if not exists train_sentences_id_ind on train_sentences(id);
