set schema 'annotation';

insert into upstream_root_did
	select
		nextval('adid') as adid
		,t1.did
		,t2.acid
		,t1.term
		,'t' as active
		,t1.effectivetime
	from annotation.cleaned_selected_descriptions_de_duped t1
	join annotation.upstream_root_cid t2
		on t1.cid = t2.cid
	left outer join (
		select t4.term from annotation.new_concepts t4
	) t5
		on t1.term = t5.term
	where t1.cid not in (select cid from annotation.snomed_cid_ignore)
	ON CONFLICT (did, term) DO NOTHING
;