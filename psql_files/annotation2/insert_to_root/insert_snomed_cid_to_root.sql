set schema 'annotation2';

insert into root_cid
	select 
		nextval('acid') as acid
		,cid
		,'t' as active
		,effectivetime
	from annotation2.cleaned_selected_descriptions_de_duped
	where cid not in (select cid from annotation2.snomed_cid_ignore)
		and term not in (select t2.term from annotation2.new_concepts t1 
			join annotation2.manual_active_desc t2
			on t1.did = t2.did)
	ON CONFLICT (cid) DO NOTHING;