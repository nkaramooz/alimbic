set schema 'annotation2';

drop table if exists custom_terms;
create table if not exists custom_terms (
	did varchar(36) not null --did is a custom uuid
	,acid varchar(36) not null
	,term varchar(400) not null
	,effectivetime timestamp not null
	,unique(acid,term)
);
create index custom_lterms_did on custom_terms(did);
create index custom_terms_acid on custom_terms(acid);
create index custom_terms_term on custom_terms(term);

