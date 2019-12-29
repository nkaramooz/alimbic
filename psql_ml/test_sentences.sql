set schema 'annotation';
drop table if exists test_sentences;


create table test_sentences as (
    select *
    from annotation.training_sentences_with_version
    order by random()
    limit 10000
);

create index if not exists condition_id_test on test_sentences(condition_id);
create index if not exists treatment_id_test on test_sentences(treatment_id);
create index if not exists label_test on test_sentences(label);
create index if not exists ver_test on test_sentences(ver);
create index if not exists id_test on test_sentences(id);
