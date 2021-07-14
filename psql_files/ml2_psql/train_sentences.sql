set schema 'ml2';
drop table if exists train_sentences;


create table train_sentences as (
    select 
    	id
    	,sentence_id
        ,sentence_tuples
    	,section_ind
    	,x_train_gen
        ,x_train_spec
    	,x_train_gen_mask
        ,x_train_spec_mask
        ,condition_acid
        ,treatment_acid
    	,label
    	,ver
    from ml2.all_training_sentences
    where rand >= 0
);

create index if not exists train_sentences_label_ind on train_sentences(label);
create index if not exists train_sentences_ver_ind on train_sentences(ver);
create index if not exists train_sentences_id_ind on train_sentences(id);
