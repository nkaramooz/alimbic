set schema 'annotation2';

insert into snomed_cid_ignore
	select
		syn_cid as cid 
	from annotation2.snomed_synonyms
	where syn_rank > ref_rank 
	ON CONFLICT (cid) DO NOTHING
;