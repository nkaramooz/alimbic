set schema 'ml2';
drop table if exists test_sentences;


create table test_sentences as (
    select 
    	id
    	,sentence_id
        ,sentence_tuples
        ,condition_acid
        ,treatment_acid
    	,section_ind
    	,x_train_gen
        ,x_train_spec
    	,x_train_gen_mask
        ,x_train_spec_mask
    	,label
    	,ver
    from ml2.all_training_sentences
    where rand > 0.99
);

create index if not exists test_sentences_label_ind on test_sentences(label);
create index if not exists test_sentences_ver_ind on test_sentences(ver);
create index if not exists test_sentences_id_ind on test_sentences(id);
