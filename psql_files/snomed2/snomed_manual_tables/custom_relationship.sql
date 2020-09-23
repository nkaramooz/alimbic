set schema 'snomed2';

drop table if exists custom_relationship cascade;
create table custom_relationship(
  id varchar(36) not null
  ,sourceid varchar(36) not null
  ,destinationid varchar(36) not null
  ,typeid varchar(18) not null
  ,active char(1) not null
  ,effectivetime timestamp not null
  ,PRIMARY KEY(id)
);