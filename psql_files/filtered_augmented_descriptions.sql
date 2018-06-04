set schema 'annotation';
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" ;


insert into filtered_augmented_descriptions

select
	distinct on (conceptid, candidate)
    conceptid
    ,candidate as term
    ,public.uuid_generate_v4() as id
from (
    select 
        conceptid
        ,pre.candidate
    from annotation.prefiltered_augmented_descriptions pre
    join (
	    select
	        count(distinct(conceptid))
	        ,candidate
	    from annotation.prefiltered_augmented_descriptions
	    group by candidate
	    having count(distinct(conceptid)) = 1 
	    ) dist 
      on pre.candidate = dist.candidate
    where conceptid || pre.candidate not in (select conceptid || term from annotation.filtered_augmented_descriptions)
  ) tb
;
