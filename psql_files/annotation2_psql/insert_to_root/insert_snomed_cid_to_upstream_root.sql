set schema 'annotation2';

insert into upstream_root_cid
	select 
		nextval('acid') as acid
		,cid
		,'t' as active
		,effectivetime
	from annotation2.cleaned_selected_descriptions_de_duped
	where cid not in (select cid from annotation2.snomed_cid_ignore)
	ON CONFLICT (cid) DO NOTHING
;