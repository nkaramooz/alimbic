set schema 'annotation2';

-- migrate new_concepts
-- insert into annotation2.new_concepts
-- 	select cid, did, term, effectivetime from annotation.new_concepts
-- ;


-- manual_snomed_cid_ignore
-- insert into annotation2.manual_snomed_ignore
-- 	select 
-- 		cid
-- 		,effectivetime
-- 	from annotation.manual_snomed_ignore
-- ;



-- migrate annotation.root_new_desc
-- insert into annotation2.root_new_desc
-- 	select 
-- 		t1.did
-- 		,t3.acid
-- 		,term
-- 		,t1.active
-- 		,t1.effectivetime
-- 	from annotation.root_new_desc t1
-- 	join annotation.upstream_root_cid t2
-- 	on t1.acid=t2.acid
-- 	join annotation2.upstream_root_cid t3
-- 	on t2.cid=t3.cid
-- ;

-- ;


-- -- migrate root_desc_inactive
-- insert into annotation2.root_desc_inactive
-- 	select 
-- 		t5.adid
-- 		,t1.term
-- 		,t1.active
-- 		,t1.effectivetime
-- 	from annotation.root_desc_inactive t1
-- 	join annotation.upstream_root_did t2
-- 	on t1.adid::int=t2.adid
-- 	join annotation.upstream_root_cid t3
-- 	on t2.acid=t3.acid
-- 	join annotation2.upstream_root_cid t4
-- 	on t3.cid=t4.cid
-- 	join annotation2.upstream_root_did t5
-- 	on t4.acid=t5.acid and t1.term=t5.term
-- ;

-- migrate inactive concepts
-- insert into annotation2.inactive_concepts
-- 	select 
-- 		t3.acid
-- 		,t1.active
-- 		,t1.effectivetime
-- 	from annotation.inactive_concepts t1
-- 	join annotation.upstream_root_cid t2
-- 	on t1.acid=t2.acid
-- 	join annotation2.upstream_root_cid t3
-- 	on t2.cid=t3.cid
-- ;

-- -- migrate acronym_override
-- insert into annotation2.acronym_override
-- select 
-- 	id
-- 	,t5.adid
-- 	,t1.is_acronym
-- 	,t1.effectivetime
-- from annotation.acronym_override t1
-- join annotation.upstream_root_did t2
-- on t1.adid::int=t2.adid
-- join annotation.upstream_root_cid t3
-- on t2.acid=t3.acid
-- join annotation2.upstream_root_cid t4
-- on t3.cid=t4.cid
-- join annotation2.upstream_root_did t5
-- on t4.acid=t5.acid and t2.term=t5.term
-- ;
