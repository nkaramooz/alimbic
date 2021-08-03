set schema 'snomed2';


insert into transitive_closure_acid
	select
	t2.acid as  child_acid
	,t3.acid as parent_acid
	from snomed2.transitive_closure_cid t1
	join annotation.downstream_root_cid t2
	on t1.child_cid = t2.cid
	join annotation.downstream_root_cid t3
	on t1.parent_cid = t3.cid
;