set schema 'ml2';
drop table if exists test_sentences;


create table test_sentences as (
    select 
    	id
    	,t1.sentence_id
        ,t1.sentence_tuples
        ,condition_acid
        -- ,t3.term as condition
        ,treatment_acid
        -- ,t4.term as treatment
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
    -- join pubmed.sentence_tuples_2 t2
    -- on t1.sentence_id = t2.sentence_id
    -- join annotation2.preferred_concept_names t3
    -- on t1.condition_acid = t3.acid
    -- join annotation2.preferred_concept_names t4
    -- on t1.treatment_acid=t4.acid
    where rand >= 0.980 
    -- and section != 'results' and section !='methods'
    --  and rand < 0.9994
);

create index if not exists test_sentences_label_ind on test_sentences(label);
create index if not exists test_sentences_ver_ind on test_sentences(ver);
create index if not exists test_sentences_id_ind on test_sentences(id);

-- insert into ml2.test_sentences
--     select
--         id
--         ,sentence_id
--         ,sentence_tuples
--         ,condition_acid
--         ,treatment_acid
--         ,og_condition_acid
--         ,og_treatment_acid
--         ,section_ind
--         ,x_train_gen
--         ,x_train_spec
--         ,x_train_gen_mask
--         ,x_train_spec_mask
--         ,label
--         ,ver
--     from ml2.all_training_sentences
--     where rand > 0.999 and rand < 0.9994 and ver=0
--     ON CONFLICT DO NOTHING;
