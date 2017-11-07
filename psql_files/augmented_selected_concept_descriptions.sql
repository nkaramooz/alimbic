set schema 'annotation';
drop table if exists augmented_selected_concept_descriptions;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" ;

create table augmented_selected_concept_descriptions as (
	select
		id as description_id
	    ,conceptid
	    ,term
	    ,'1'::text as active
	    ,now() as effectivetime
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
			description_id as id
		    ,conceptid
		    ,term
		from annotation.description_whitelist
		) tb
	where id not in (select id from annotation.description_blacklist)
);

create index ascd_conceptid_ind on augmented_selected_concept_descriptions(conceptid);
create index ascd_description_id_ind on augmented_selected_concept_descriptions(description_id);
create index ascd_term_ind on augmented_selected_concept_descriptions(term);
create index ascd_active_ind on augmented_selected_concept_descriptions(active);
create index ascd_effectivetime_ind on augmented_selected_concept_descriptions(effectivetime);