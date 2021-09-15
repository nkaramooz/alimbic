set schema 'annotation2';

drop table if exists mdc_final;

create table mdc_final (
  acid varchar(40)
  ,title varchar(400)
  ,description varchar(400)
  ,url varchar(400)
);


insert into mdc_final 
	select
		acid
		,title
		,t1.desc as description
		,url
	from annotation2.mdc_staging t1
	join annotation2.concept_types t2
	on t1.acid=t2.root_acid
	join snomed2.transitive_closure_acid t3
	on t1.acid=t3.child_acid
	where t2.rel_type='condition' or t2.rel_type='symptom' and t2.active=1
	
;