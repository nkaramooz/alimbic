set schema 'annotation';
drop table if exists test_sentences_v2;


create table test_sentences_v2 as (
    select *
    from annotation.training_sentences_with_version_v2
    order by random()
    limit 100000
);

create index if not exists condition_id_test_v2 on test_sentences_v2(condition_id);
create index if not exists treatment_id_test_v2 on test_sentences_v2(treatment_id);
create index if not exists label_test_v2 on test_sentences_v2(label);
create index if not exists ver_gen_test_v2 on test_sentences_v2(ver_gen);
create index if not exists id_test_v2 on test_sentences_v2(id);
