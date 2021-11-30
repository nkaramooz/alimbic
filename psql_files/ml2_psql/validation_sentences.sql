set schema 'ml2';
drop table if exists validation_sentences;


create table validation_sentences as (
    select 
    	id
    	,t1.sentence_id
        ,t1.sentence_tuples
        ,condition_acid
        ,t2.term as condition
        ,treatment_acid
        ,t3.term as treatment
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
    join annotation2.preferred_concept_names t2
    on t1.condition_acid = t2.acid
    join annotation2.preferred_concept_names t3
    on t1.treatment_acid = t3.acid
    join pubmed.sentence_tuples_2 t4
    on t1.sentence_id = t4.sentence_id
    where rand > 0.985 and rand < 0.995 and section != 'results' and section !='methods'

);

create index if not exists validation_sentences_label_ind on validation_sentences(label);
create index if not exists validation_sentences_ver_ind on validation_sentences(ver);
create index if not exists validation_sentences_id_ind on validation_sentences(id);