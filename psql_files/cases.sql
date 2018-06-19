set schema 'vancocalc';
drop table if exists cases;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

create table cases (
	cid uuid,
	uid uuid,
	casename varchar(40),
	active integer,
	effectivetime timestamp
);