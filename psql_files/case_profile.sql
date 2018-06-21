set schema 'vancocalc';
drop table if exists case_profile;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

create table case_profile (
	eid uuid,
	cid uuid,
	type varchar(40),
	value decimal,
	str_value varchar(40),
	active integer,
	effectivetime timestamp
);
