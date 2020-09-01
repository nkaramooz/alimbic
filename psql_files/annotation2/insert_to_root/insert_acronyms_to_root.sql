set schema 'annotation2';

insert into root_did
	select 
		nextval('adid') as adid
		,t1.did
		,t2.acid
		,t1.candidate as term
		,'t' as active
		,t1.effectivetime
	from annotation2.acronym_augmented_descriptions t1
	join annotation2.root_cid t2
		on t1.cid = t2.cid
	left outer join annotation2.manual_inactive_desc t3
		on t1.cid = t3.cid and t1.candidate = t3.term
	ON CONFLICT (did, term) DO NOTHING
;


