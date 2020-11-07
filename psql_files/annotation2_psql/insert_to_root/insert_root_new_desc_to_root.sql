set schema 'annotation2';

insert into upstream_root_did
	select
		nextval('adid') as adid
		,t2.did
		,t2.acid
		,t2.term
		,t2.active
		,t2.effectivetime
	from (
		select 
			did
			,acid
			,term
			,active
			,effectivetime
		from (
			select 
				did
				,acid
				,term
				,active
				,effectivetime
				,row_number () over (partition by did order by effectivetime desc) as row_num
			from annotation2.root_new_desc
			) t1
		where row_num = 1
	) t2
	ON CONFLICT (acid, term) DO NOTHING
;