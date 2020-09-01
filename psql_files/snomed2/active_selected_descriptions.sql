set schema 'snomed2';

insert into active_selected_descriptions
	select 
		tb2.did
		,tb1.cid
		,tb2.term
		,now() as effectivetime
	from snomed2.active_selected_concepts tb1
	join snomed2.active_descriptions tb2
		on tb1.cid = tb2.cid
	ON CONFLICT DO NOTHING;