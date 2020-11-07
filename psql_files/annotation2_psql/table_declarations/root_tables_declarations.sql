set schema 'annotation2';


drop table if exists upstream_root_cid cascade;
create table upstream_root_cid(
	acid integer not null
	,cid varchar(36) not null
	,active boolean not null
	,effectivetime timestamp not null
	,unique(cid)
);
create index concurrently upstream_root_cid_acid_ind on upstream_root_cid(acid);
create index concurrently upstream_root_cid_cid_ind on upstream_root_cid(cid);



drop table if exists upstream_root_did cascade;
create table upstream_root_did(
	adid integer not null
	,did varchar(36) not null
	,acid integer not null
	,term varchar(400) not null
	,active boolean not null
	,effectivetime timestamp not null
	,unique(did, term)
	,unique(acid, term)
);
create index concurrently upstream_root_did_adid_ind on upstream_root_did(adid);
create index concurrently upstream_root_did_did_ind on upstream_root_did(did);



drop table if exists root_new_desc cascade;
create table root_new_desc(
	did varchar(36) not null
	,acid integer not null
	,term varchar(400) not null
	,active boolean not null
	,effectivetime timestamp not null
);
create index concurrently root_new_desc_adid_ind on root_new_desc(did);
create index concurrently root_new_desc_acid_ind on root_new_desc(acid);
create index concurrently root_new_desc_term_ind on root_new_desc(term);


drop table if exists root_desc_inactive cascade;
create table root_desc_inactive(
	adid varchar(40) not null
	,term varchar(400) not null
	,active boolean not null
	,effectivetime timestamp not null
);
create index concurrently root_desc_inactive_adid_ind on root_desc_inactive(adid);
create index concurrently root_desc_inactive_term_ind on root_desc_inactive(term);


drop table if exists inactive_concepts cascade;
create table inactive_concepts(
	acid varchar(40) not null
	,active boolean not null
	,effectivetime timestamp not null
);
create index concurrently inactive_concepts_acid_ind on inactive_concepts(acid);


drop table if exists annotation2.new_concepts;
create table new_concepts(
	cid varchar(36) not null
	,did varchar(36) not null
	,term varchar(400) not null
	,effectivetime timestamp not null
	,PRIMARY KEY (cid)
);
create index concurrently new_concepts_cid_ind on new_concepts(cid);
create index concurrently new_concepts_did_ind on new_concepts(did);
create index concurrently new_concepts_term_ind on new_concepts(term);


drop table if exists acronym_override;
create table acronym_override(
	id varchar(36) not null
	,adid varchar(36) not null
	,is_acronym boolean not null
	,effectivetime timestamp not null
	,PRIMARY KEY (id)
);
create index concurrently acronym_override_adid_ind on acronym_override(adid);


create table if not exists manual_snomed_ignore (
	cid varchar(36) not null
	,effectivetime timestamp not null
);
create index concurrently manual_snomed_ignore_cid_ind on manual_snomed_ignore(cid);

create sequence annotation2.acid increment by 1 start 1 owned by annotation2.upstream_root_cid.acid;
create sequence annotation2.adid increment by 1 start 1 owned by annotation2.upstream_root_did.adid;