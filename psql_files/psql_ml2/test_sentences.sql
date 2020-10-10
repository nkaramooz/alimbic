set schema 'annotation';
drop table if exists test_sentences_v2;


create table test_sentences_v2 as (
    select 
    	id,
    	sentence_id, 
    	x_train_gen, 
    	label,
    	0 as ver
    from annotation.root_training_sentences
    order by random()
    limit 100000
);

create index if not exists label_test_v2 on test_sentences_v2(label);
create index if not exists ver_gen_test_v2 on test_sentences_v2(ver);
create index if not exists id_test_v2 on test_sentences_v2(id);
