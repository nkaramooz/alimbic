set schema 'snomed2';

insert into selected_concepts 
	select 
		distinct (subtypeid) as cid
		,NOW() as effectivetime
	from snomed2.curr_transitive_closure_f
	where supertypeid in ('404684003', '123037004', '363787002', 
		'410607006', '373873005', '71388002', '105590001', 
		'362981000', '48176007', '49062001', '260787004')
	ON CONFLICT (cid) DO NOTHING;