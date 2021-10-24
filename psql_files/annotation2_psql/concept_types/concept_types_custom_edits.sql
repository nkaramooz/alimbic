-- inactivate or activate certain branches of graph


-- insert into annotation2.concept_types_app
-- 	VALUES 
-- 	('98381', 'diagnostic', 0, now())
-- ;

-- insert into annotation2.concept_types_app
-- 	select 
-- 		child_acid as acid
-- 		,'diagnostic' as rel_type
-- 		,0 as active
-- 		,now() as effectivetime
-- 	from snomed2.transitive_closure_acid 
-- 	where parent_acid='98381'
-- ;





insert into annotation2.concept_types_app
	VALUES 
	('288161', 'treatment', 0, now())
;

insert into annotation2.concept_types_app
	select 
		child_acid as acid
		,'treatment' as rel_type
		,0 as active
		,now() as effectivetime
	from snomed2.transitive_closure_acid 
	where parent_acid='288161'
;




-- insert into annotation2.concept_types_app
-- 	VALUES 
-- 	('178715', 'condition', 0, now())
-- ;

-- insert into annotation2.concept_types_app
-- 	select 
-- 		child_acid as acid
-- 		,'condition' as rel_type
-- 		,0 as active
-- 		,now() as effectivetime
-- 	from snomed2.transitive_closure_acid 
-- 	where parent_acid='340422'
-- ;






-- insert into annotation2.concept_types_app
-- 	VALUES 
-- 	('515062', 'symptom', 0, now())
-- ;


-- insert into annotation2.concept_types_app
-- 	select 
-- 		child_acid as acid
-- 		,'symptom' as rel_type
-- 		,0 as active
-- 		,now() as effectivetime
-- 	from snomed2.transitive_closure_acid 
-- 	where parent_acid='515062'
-- ;




-- insert into annotation.concept_types_app
-- 	VALUES 
-- 	('178715', 'anatomy', 0, now())
-- ;

-- insert into annotation2.concept_types_app
-- 	select 
-- 		child_acid as acid
-- 		,'anatomy' as rel_type
-- 		,0 as active
-- 		,now() as effectivetime
-- 	from snomed2.transitive_closure_acid 
-- 	where parent_acid='21100'
-- ;




