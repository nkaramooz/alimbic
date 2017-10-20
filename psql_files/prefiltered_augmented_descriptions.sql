set schema 'annotation';

drop table if exists prefiltered_augmented_descriptions;
create table prefiltered_augmented_descriptions as (
    select conceptid, term, candidate
    from (
        select conceptid, term, (regexp_matches(term, '(.*?) - '))[1] as candidate
        from annotation.active_selected_concept_descriptions
        ) tb 
    where (candidate not in (select term from annotation.active_selected_concept_descriptions))
);