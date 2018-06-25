set schema 'vancocalc';
drop table if exists trough;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

create table trough (
	tid uuid,
	cid uuid,
	trough decimal,
	beforeDoseNum integer,
	active integer,
	effectivetime timestamp
);