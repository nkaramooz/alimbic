set schema 'annotation2';

insert into root_cid
	select 
		nextval('acid') as acid
		,cid
		,'t' as active
		,effectivetime
	from annotation2.new_concepts
	ON CONFLICT (cid) DO NOTHING;