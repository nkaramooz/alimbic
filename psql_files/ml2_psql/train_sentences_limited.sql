set schema 'ml2';
drop table if exists train_sentences_limited;


create table train_sentences_limited as (
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
    	,0 as ver
    from ml2.train_sentences t1
    join annotation2.concept_types t2
    on t1.treatment_acid=t2.root_acid
    where t2.rel_type='treatment'
);

create index if not exists train_sentences_limited_label_ind on train_sentences_limited(label);
create index if not exists train_sentences_limited_ver_ind on train_sentences_limited(ver);
create index if not exists train_sentences_limited_id_ind on train_sentences_limited(id);
