set schema 'annotation';
-- since snomed has occasional concepts with a different primary name but similar description names
-- in those cases, will choose the one with the lowest cid number
insert into snomed_cid_ignore
	select
	syn_cid as cid 
	from annotation.snomed_synonyms
	where syn_rank > ref_rank or (syn_rank = ref_rank and syn_cid < ref_cid)

	union 

	select
	cid
	from annotation.manual_snomed_ignore
	
	ON CONFLICT (cid) DO NOTHING
;