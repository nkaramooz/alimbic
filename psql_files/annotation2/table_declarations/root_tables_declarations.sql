set schema 'annotation2';


drop table if exists root_cid cascade;
create table root_cid(
	acid integer not null
	,cid varchar(36) not null
	,active boolean not null
	,effectivetime timestamp not null
	,unique(cid)
);
create index concurrently root_cid_acid_ind on root_cid(acid);
create index concurrently root_cid_cid_ind on root_cid(cid);



drop table if exists root_did cascade;
create table root_did(
	adid integer not null
	,did varchar(36) not null
	,acid integer not null
	,term varchar(400) not null
	,active boolean not null
	,effectivetime timestamp not null
	,unique(did, term)
);
create index concurrently root_did_adid_ind on root_did(adid);
create index concurrently root_did_did_ind on root_did(did);



create sequence acid increment by 1 start 1;
create sequence adid increment by 1 start 1;