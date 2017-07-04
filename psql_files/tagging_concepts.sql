set schema 'snomed';

drop table if exists tagging_concepts;
create table tagging_concepts as (
	select 
		distinct (supertypeId) as conceptid
	from snomed.curr_transitive_closure_f
	where subtypeId in ('404684003', '123037004', '363787002', '410607006', '373873005', '71388002')
);

