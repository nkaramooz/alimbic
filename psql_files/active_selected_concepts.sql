set schema 'annotation';

drop table if exists active_selected_concepts;
create table active_selected_concepts as (
	select
		conceptid
	from annotation.selected_concepts s
	join (
		select id, active 
		from (
				select id, active, row_number () over 
					(partition by id order by effectivetime desc) as row_num
				from snomed.curr_concept_f
		) tb 
		where row_num = 1 and active = '1'
	) a 
	on s.conceptid = a.id
);

