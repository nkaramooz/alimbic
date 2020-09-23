set schema 'snomed2';

drop table if exists transitive_closure_cid cascade;
create table transitive_closure_cid (
	child_cid varchar(36) not null
	,parent_cid varchar(36)
);
create index transitive_closure_cid_child_cid_ind on transitive_closure_cid(child_cid);
create index transitive_closure_cid_parent_cid_ind on transitive_closure_cid(parent_cid);


drop table if exists transitive_closure_acid cascade;
create table transitive_closure_acid (
	child_cid varchar(36) not null
	,parent_cid varchar(36)
);
create index transitive_closure_acid_child_cid_ind on transitive_closure_acid(child_cid);
create index transitive_closure_acid_parent_cid_ind on transitive_closure_acid(parent_cid);