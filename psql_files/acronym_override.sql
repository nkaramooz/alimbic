set schema 'annotation';
drop table if exists acronym_override;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

create table acronym_override(
	id uuid,
	description_id varchar(40),
	is_acronym integer,
	effectivetime timestamp
);
-- 0 = not acronym
-- 1 = acronym