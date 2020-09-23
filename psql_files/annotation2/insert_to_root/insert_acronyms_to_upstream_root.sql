set schema 'annotation2';

insert into upstream_root_did
	select 
		nextval('adid') as adid
		,t1.did
		,t2.acid
		,t1.candidate as term
		,'t' as active
		,t1.effectivetime
	from annotation2.acronym_augmented_descriptions t1
	join annotation2.upstream_root_cid t2
		on t1.cid = t2.cid
	ON CONFLICT (acid, term) DO NOTHING
;


