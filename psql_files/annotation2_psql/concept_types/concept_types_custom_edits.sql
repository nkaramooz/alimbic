-- inactivate or activate certain branches of graph

insert into annotation2.concept_types_app
	select 
		child_acid as acid
		,'diagnostic' as rel_type
		,0 as active
		,now() as effectivetime
	from snomed2.transitive_closure_acid 
	where parent_acid='451490'
;


insert into annotation2.concept_types_app
	VALUES 
	('451490', 'diagnostic', 0, now())
;


-- insert into annotation2.concept_types_app
-- 	select 
-- 		child_acid as acid
-- 		,'treatment' as rel_type
-- 		,0 as active
-- 		,now() as effectivetime
-- 	from snomed2.transitive_closure_acid 
-- 	where parent_acid='290045'
-- ;


-- insert into annotation2.concept_types_app
-- 	VALUES 
-- 	('290045', 'treatment', 0, now())
-- ;
