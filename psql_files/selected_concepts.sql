set schema 'annotation';

drop table if exists selected_concepts;
create table selected_concepts as (
	select 
		distinct (supertypeId) as conceptid
	from snomed.curr_transitive_closure_f
	where subtypeId in ('404684003', '123037004', '363787002', 
		'410607006', '373873005', '71388002', '105590001', '362981000')
);

