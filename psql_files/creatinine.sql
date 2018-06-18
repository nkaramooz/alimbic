set schema 'vancocalc';
drop table if exists creatinine;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" ;

create table creatinine (
	crid uuid,
	cid uuid,
	creatinine decimal,
	crcl decimal,
	active integer,
	effectivetime timestamp
);