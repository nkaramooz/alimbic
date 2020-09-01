set schema 'annotation2';

insert into root_did
	select
		nextval('adid') as adid
		,t1.did
		,t2.acid
		,t1.term
		,'t' as active
		,t1.effectivetime
	from annotation2.cleaned_selected_descriptions_de_duped t1
	join annotation2.root_cid t2
		on t1.cid = t2.cid
	left outer join annotation2.manual_inactive_desc t3
		on t1.cid = t3.cid and t1.term = t3.term
	left outer join (
		select t4.term from annotation2.manual_active_desc t4
		right join annotation2.new_concepts t5
			on t4.cid = t5.cid
			) t6
		on t1.term = t6.term
	where t1.cid not in (select cid from annotation2.snomed_cid_ignore)
	ON CONFLICT (did, term) DO NOTHING
;