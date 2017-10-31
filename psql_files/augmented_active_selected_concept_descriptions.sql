set schema 'annotation';
drop table if exists augmented_active_selected_concept_descriptions;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" ;

create table augmented_active_selected_concept_descriptions as (
	select
		id
	    ,conceptid
	    ,term
	from (
		select 
			distinct on (conceptid, term)
			id
		  	,conceptid
		    ,term
		from annotation.active_cleaned_selected_concept_descriptions

		union

		select
			id::text
		    ,conceptid
		    ,term
		from annotation.filtered_augmented_descriptions

		union

		select 
			public.uuid_generate_v4()::text as id
		    ,conceptid
		    ,term
		from annotation.description_whitelist
		) tb
	where id not in (select id from annotation.description_id_blacklist)
);