set schema 'snomed2';

drop table if exists full_relationship cascade;
create table full_relationship (
	sourceid varchar(36) not null
	,destinationid varchar(36)
	,typeid varchar(18) not null
	,active char(1) not null
);

-- the below case is a bad idea
insert into full_relationship
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
		,effectivetime
		from (
			select
			sourceid
			,destinationid
			,typeid
			,active
			,effectivetime
			,row_number() over (partition by sourceid, destinationid order by effectivetime desc) as rn_num
			from (
				select 
				sourceid
				,destinationid
				,typeid
				,active
				,effectivetime::timestamp
				from snomed2.active_snomed_relationships

				union

				select  
				sourceid
				,destinationid
				,typeid
				,active
				,effectivetime
				from snomed2.custom_relationship
				) t4
			) t5
		where rn_num = 1
		) t6
	where active='1'
;