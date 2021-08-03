set schema 'annotation';

insert into upstream_root_did
	select 
		nextval('adid') as adid
		,t1.did
		,t1.acid::int
		,t1.term
		,'t' as active
		,t1.effectivetime
	from annotation.custom_terms t1
	ON CONFLICT (acid, term) DO NOTHING
;


