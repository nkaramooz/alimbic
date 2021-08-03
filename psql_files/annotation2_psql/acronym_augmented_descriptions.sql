set schema 'annotation';

insert into acronym_augmented_descriptions
    select cid, did, term, candidate, now() as effectivetime
    from (
        select cid, did, term, (regexp_matches(term, '(.*?) - '))[1] as candidate
        from annotation.cleaned_selected_descriptions_de_duped
        where cid in (
            select child_cid
            from snomed2.transitive_closure_cid 
            where parent_cid in ('410607006', '123037004', '404684003', '71388002', '105590001'))
            and term not ilike '%on exam%' and term not ilike '%-%-%' and term not ilike '%O/E%'
        ) tb 
    where (candidate not in (select term from annotation.cleaned_selected_descriptions_de_duped))
    	and cid not in (select cid from annotation.snomed_cid_ignore)
        and upper(candidate) = candidate
	ON CONFLICT DO NOTHING;