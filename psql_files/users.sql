set schema 'vancocalc';
drop table if exists users;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" ;

create table users (
	uid uuid,
	username varchar(40),
	active integer,
	effectivetime timestamp
);