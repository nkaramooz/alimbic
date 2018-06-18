set schema 'vancocalc';
drop table if exists weight;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" ;

create table weight (
	wtid uuid,
	cid uuid,
	weight decimal,
	dosingWeight decimal,
	active integer,
	effectivetime timestamp
);