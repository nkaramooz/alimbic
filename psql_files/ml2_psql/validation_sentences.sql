set schema 'ml2';
drop table if exists validation_sentences;


create table validation_sentences as (
    select 
    	id
    	,sentence_id
        ,sentence_tuples
        ,condition_acid
        ,treatment_acid
        ,og_condition_acid
        ,og_treatment_acid
    	,section_ind
    	,x_train_gen
        ,x_train_spec
    	,x_train_gen_mask
        ,x_train_spec_mask
    	,label
    	,ver
    from ml2.all_training_sentences
    where rand > 0.985 and rand < 0.995

);

create index if not exists validation_sentences_label_ind on validation_sentences(label);
create index if not exists validation_sentences_ver_ind on validation_sentences(ver);
create index if not exists validation_sentences_id_ind on validation_sentences(id);