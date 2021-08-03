set schema 'snomed2';

insert into active_selected_concepts 
	select
		cid
		,now() as effectivetime
	from snomed2.selected_concepts s
	join (
		select id, active 
		from (
				select id, active, row_number () over 
					(partition by id order by effectivetime desc) as row_num
				from snomed2.curr_concept_f
		) tb 
		where row_num = 1 and active = '1'
	) a 
	on s.cid = a.id
	ON CONFLICT (cid) DO NOTHING;

