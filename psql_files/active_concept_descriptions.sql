set schema 'annotation';

drop table if exists active_concept_descriptions;
create table active_concept_descriptions as (
    select *
    from (
        select 
            id
            ,effectivetime
            ,row_number () over (partition by id order by effectivetime desc) as row_num
            ,term
            ,active
            ,conceptid
            ,typeid    
        from snomed.curr_description_f
      ) tb
    where row_num = 1 and active='1'
);