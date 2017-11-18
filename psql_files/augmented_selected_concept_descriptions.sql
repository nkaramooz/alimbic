set schema 'annotation';
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" ;

insert into augmented_selected_concept_descriptions
	select
		id as description_id
	    ,conceptid
	    ,term
	    ,active
	    ,case when effectivetime is null then now() else effectivetime end as effectivetime
	from (
		select 
			distinct on (conceptid, term)
			id
		  	,conceptid
		    ,term
		    ,'1'::text as active
		    ,null::timestamp as effectivetime
		from annotation.active_cleaned_selected_concept_descriptions

		union

		select
			id::text
		    ,conceptid
		    ,term
		    ,'1'::text as active
		    ,null::timestamp as effectivetime
		from annotation.filtered_augmented_descriptions

		union

		select 
			description_id as id
		    ,conceptid
		    ,term
		    ,'1'::text as active
		    ,null::timestamp as effectivetime
		from annotation.description_whitelist

		union 

		select 
			description_id as id
			,conceptid 
			,description as term 
			,'1'::text as active
			,effectivetime
		from annotation.new_concepts

		union

		select 
			description_id as id
			,conceptid 
			,term 
			,'0'::text as active
			,null::timestamp as effectivetime 			
		from annotation.description_blacklist 
		) tb 
	where id || active || effectivetime  not in (select description_id || active || effectivetime from annotation.augmented_selected_concept_descriptions);

create index if not exists ascd_conceptid_ind on augmented_selected_concept_descriptions(conceptid);
create index if not exists ascd_description_id_ind on augmented_selected_concept_descriptions(description_id);
create index if not exists ascd_term_ind on augmented_selected_concept_descriptions(term);
create index if not exists ascd_active_ind on augmented_selected_concept_descriptions(active);
create index if not exists ascd_effectivetime_ind on augmented_selected_concept_descriptions(effectivetime);