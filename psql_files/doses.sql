set schema 'vancocalc';
drop table if exists doses;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

create table doses (
	did uuid,
	cid uuid,
	dose integer,
	freqIndex integer,
	freqString varchar(40),
	active integer,
	effectivetime timestamp
);