set schema 'annotation2';

insert into upstream_root_cid
	select 
		nextval('acid') as acid
		,cid
		,'t' as active
		,effectivetime
	from annotation2.new_concepts
	ON CONFLICT (cid) DO NOTHING;

insert into upstream_root_did
	select 
		nextval('adid')
		,t1.did
		,t2.acid
		,t1.term
		,'t' as active
		,t1.effectivetime
	from annotation2.new_concepts t1
	left join annotation2.upstream_root_cid t2
		on t1.cid = t2.cid
	ON CONFLICT (did, term) DO NOTHING
;