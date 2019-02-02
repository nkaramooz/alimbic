set schema 'annotation';

drop table if exists preferred_concept_names;
create table preferred_concept_names as (
	select 
		conceptid
		,term
	from (
		select 
			length(term)
    		,term
    		,conceptid
    		,row_number () over (partition by conceptid order by length(term) desc) as row_num
		from annotation.augmented_selected_concept_descriptions
		where term not ilike '%(%' and term not ilike '%-%' and term not ilike '%product%') tb
	where row_num=1
);

create index if not exists concept_names_cid on preferred_concept_names(conceptid);
create index if not exists concept_names on preferred_concept_names(term);