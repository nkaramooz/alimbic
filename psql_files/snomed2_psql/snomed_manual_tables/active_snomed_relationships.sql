set schema 'snomed2';

drop table if exists active_snomed_relationships cascade;
create table active_snomed_relationships(
  sourceid varchar(36) not null
  ,destinationid varchar(36) not null
  ,typeid varchar(18) not null
  ,active char(1) not null
  ,effectivetime timestamp not null
 );


insert into active_snomed_relationships
	select
		sourceid
		,destinationid
		,typeid
		,active
		,'1900-09-08 19:58:14.190442' as effectivetime
	from (
		select
			sourceid
			,destinationid
			,typeid
			,active
		from (
			select 
				sourceid
				,destinationid
				,typeid
				,active
				,row_number() over (partition by sourceid, destinationid, typeid order by effectivetime desc) as rn_num
			from snomed2.curr_relationship_f
		) t1
		where rn_num = 1
	) t2
	where active='1'
;