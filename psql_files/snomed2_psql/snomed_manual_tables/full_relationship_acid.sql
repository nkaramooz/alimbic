set schema 'snomed2';


drop table if exists full_relationship_acid cascade;
create table full_relationship_acid(
  source_acid varchar(36) not null
  ,destination_acid varchar(36) not null
  ,typeid varchar(18) not null
  ,active char(1) not null
);

create index full_relationship_acid_source_acid_ind on full_relationship_acid(source_acid);
create index full_relationship_acid_destination_acid_ind on full_relationship_acid(destination_acid);
create index full_relationship_acid_active_ind on full_relationship_acid(active);
create index full_relationship_acid_typeid_ind on full_relationship_acid(typeid);

insert into full_relationship_acid
	select
	t2.acid as source_acid
	,t3.acid as destionation_acid
	,t1.typeid
	,t1.active
	from snomed2.full_relationship t1
	join annotation2.upstream_root_cid t2
	on t1.sourceid = t2.cid
	join annotation2.upstream_root_cid t3
	on t1.destinationid = t3.cid




