-- inactivate or activate certain branches of graph

insert into annotation2.concept_types_app
	select 
		child_acid as acid
		,'diagnostic' as rel_type
		,0 as active
		,now() as effectivetime
	from snomed2.transitive_closure_acid 
	where parent_acid='14122'
;


insert into annotation2.concept_types_app
	VALUES 
	('14122', 'diagnostic', 0, now())
;


-- insert into annotation2.concept_types_app
-- 	select 
-- 		child_acid as acid
-- 		,'treatment' as rel_type
-- 		,0 as active
-- 		,now() as effectivetime
-- 	from snomed2.transitive_closure_acid 
-- 	where parent_acid='345359'
-- ;


-- insert into annotation2.concept_types_app
-- 	VALUES 
-- 	('345359', 'treatment', 0, now())
-- ;

-- insert into annotation2.concept_types_app
-- 	select 
-- 		child_acid as acid
-- 		,'condition' as rel_type
-- 		,0 as active
-- 		,now() as effectivetime
-- 	from snomed2.transitive_closure_acid 
-- 	where parent_acid='218363'
-- ;


-- insert into annotation2.concept_types_app
-- 	VALUES 
-- 	('218363', 'condition', 0, now())
-- ;

-- insert into annotation2.concept_types_app
-- 	select 
-- 		child_acid as acid
-- 		,'symptom' as rel_type
-- 		,0 as active
-- 		,now() as effectivetime
-- 	from snomed2.transitive_closure_acid 
-- 	where parent_acid='218363'
-- ;


-- insert into annotation2.concept_types_app
-- 	VALUES 
-- 	('218363', 'symptom', 0, now())
-- ;


-- insert into annotation2.concept_types_app
-- 	select 
-- 		child_acid as acid
-- 		,'anatomy' as rel_type
-- 		,0 as active
-- 		,now() as effectivetime
-- 	from snomed2.transitive_closure_acid 
-- 	where parent_acid='107780'
-- ;


-- insert into annotation2.concept_types_app
-- 	VALUES 
-- 	('107780', 'anatomy', 0, now())
-- ;

