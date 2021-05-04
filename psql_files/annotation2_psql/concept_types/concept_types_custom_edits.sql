insert into annotation2.concept_types_app
	select 
		child_acid as acid
		,'symptom' as rel_type
		,0 as active
		,now() as effectivetime
	from snomed2.transitive_closure_acid 
	where parent_acid='510990'
;