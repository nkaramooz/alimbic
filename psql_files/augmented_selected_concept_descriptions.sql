set schema 'annotation';

insert into augmented_selected_concept_descriptions
	select
		id as description_id
	    ,conceptid
	    ,term
	    ,active
	    ,case when effectivetime is null then now() else effectivetime end as effectivetime
	from annotation.prelim_augmented_selected_concept_descriptions
 	where not exists (
 		select description_id from annotation.augmented_selected_concept_descriptions
        where annotation.prelim_augmented_selected_concept_descriptions.id = annotation.augmented_selected_concept_descriptions.description_id
        and annotation.prelim_augmented_selected_concept_descriptions.active = annotation.augmented_selected_concept_descriptions.active           
        );

create index if not exists ascd_conceptid_ind on augmented_selected_concept_descriptions(conceptid);
create index if not exists ascd_description_id_ind on augmented_selected_concept_descriptions(description_id);
create index if not exists ascd_term_ind on augmented_selected_concept_descriptions(term);
create index if not exists ascd_active_ind on augmented_selected_concept_descriptions(active);
create index if not exists ascd_effectivetime_ind on augmented_selected_concept_descriptions(effectivetime);