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
	join snomed2.transitive_closure_acid t2
	on t1.acid=t2.child_acid
	where t2.parent_acid in ('11220', '346270', '165831', '250976', '241259')

;