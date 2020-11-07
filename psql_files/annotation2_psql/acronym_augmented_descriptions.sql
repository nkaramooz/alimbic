set schema 'annotation2';

insert into acronym_augmented_descriptions
    select cid, did, term, candidate, now() as effectivetime
    from (
        select cid, did, term, (regexp_matches(term, '(.*?) - '))[1] as candidate
        from annotation2.cleaned_selected_descriptions_de_duped
        ) tb 
    where (candidate not in (select term from annotation2.cleaned_selected_descriptions_de_duped))
    	and cid not in (select cid from annotation2.snomed_cid_ignore)
	ON CONFLICT DO NOTHING;