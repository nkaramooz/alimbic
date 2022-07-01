set schema 'ml2';
-- drop table if exists validation_sentences;


-- create table validation_sentences as (
--     select 
--     	id
--     	,t1.sentence_id
--         ,t1.sentence_tuples
--         ,condition_acid
--         ,treatment_acid
--         ,og_condition_acid
--         ,og_treatment_acid
--     	,t1.section_ind
--     	,x_train_gen
--         ,x_train_spec
--     	,x_train_gen_mask
--         ,x_train_spec_mask
--     	,label
--     	,t1.ver
--     from ml2.all_training_sentences t1
--     where rand > 0.98 and rand < 0.99 

-- );

insert into ml2.validation_sentences
    select
        id
        ,t1.sentence_id
        ,t1.sentence_tuples
        ,condition_acid
        ,treatment_acid
        ,og_condition_acid
        ,og_treatment_acid
        ,t1.section_ind
        ,x_train_gen
        ,x_train_spec
        ,x_train_gen_mask
        ,x_train_spec_mask
        ,label
        ,t1.ver
    from ml2.all_training_sentences t1
    where ver=0 and rand > 0.98 and rand < 0.99
    ON CONFLICT DO NOTHING;

-- create index if not exists validation_sentences_label_ind on validation_sentences(label);
-- create index if not exists validation_sentences_ver_ind on validation_sentences(ver);
-- create index if not exists validation_sentences_id_ind on validation_sentences(id);