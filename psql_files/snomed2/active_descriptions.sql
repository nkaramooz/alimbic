set schema 'snomed2';

insert into active_descriptions
    select 
        did
        ,effectivetime
        ,term
        ,cid
        ,typeid
    from (
        select 
            id as did
            ,now() as effectivetime
            ,row_number () over (partition by id order by effectivetime desc) as row_num
            ,term
            ,active
            ,conceptid as cid
            ,typeid    
        from snomed2.curr_description_f
      ) tb
    where row_num = 1 and active='1'
    ON CONFLICT (did) DO NOTHING;