-- insert into annotation2.inactive_concepts
-- 	select
-- 		t2.acid::int
-- 		,'f'
-- 		,now()
-- 	from annotation2.downstream_root_did t1
-- 	join annotation2.downstream_root_did t2
-- 	on t1.term || ' hemisuccinate' = t2.term
-- 		and t1.acid != t2.acid
-- ;
