set schema 'annotation2';

-- migrate new_concepts
-- insert into annotation2.new_concepts
-- 	select conceptid as cid, description_id as did, description as term, effectivetime from annotation.new_concepts
-- ;


-- manual_snomed_cid_ignore
-- insert into annotation2.manual_snomed_ignore
-- 	select 
-- 		subtypeid
-- 		,now()
-- 	from snomed2.curr_transitive_closure_f
-- 	where supertypeid = '422096002'
-- ;



-- migrate description_whitelist
insert into annotation2.root_new_desc
	select
	did
	,t4.acid
	,term
	,'t' as active
	,now()
	from (
		select 
		t1.description_id as did
		,case when syn_rank < ref_rank then syn_cid when syn_rank=ref_rank and syn_cid < ref_cid then syn_cid else t1.conceptid end as cid 
		,t1.term
		from annotation.description_whitelist t1
		left join (select distinct on (ref_cid, ref_rank, syn_cid, syn_rank) ref_cid, ref_rank, syn_cid, syn_rank from annotation2.snomed_synonyms) t2
		on t1.conceptid = t2.ref_cid
		) t3
	left join annotation2.upstream_root_cid t4
	on t3.cid = t4.cid
	where acid is not null
;


-- migrate description_blacklist
insert into annotation2.root_desc_inactive
select
	t5.adid
	,t3.term
	,'t' as active
	,now()
	from (
		select 
		t1.description_id as did
		,case when syn_rank < ref_rank then syn_cid when syn_rank=ref_rank and syn_cid < ref_cid then syn_cid else t1.conceptid end as cid 
		,t1.term
		from annotation.description_blacklist t1
		left join (select distinct on (ref_cid, ref_rank, syn_cid, syn_rank) ref_cid, ref_rank, syn_cid, syn_rank from annotation2.snomed_synonyms) t2
		on t1.conceptid = t2.ref_cid
		) t3
	left join annotation2.upstream_root_cid t4
		on t3.cid = t4.cid
	left join annotation2.upstream_root_did t5
	on t3.did = t5.did
	where adid is not null
;

-- migrate acronym_override
insert into annotation2.acronym_override

	select
		t1.id
		,t2.adid
		,case when t1.is_acronym = 0 then false else true end as is_acronym
		,t1.effectivetime
	from annotation.acronym_override t1
	join annotation2.upstream_root_did t2
	on t1.description_id = t2.did
;

-- make sure to rerun update


