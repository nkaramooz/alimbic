set schema 'snomed2';


drop table if exists selected_concepts cascade;
create table selected_concepts(
  cid varchar(18) not null
  ,effectivetime timestamp not null
  ,PRIMARY KEY(cid)
);
create index concurrently selected_concepts_cid_ind on selected_concepts(cid);


drop table if exists active_selected_concepts cascade;
create table active_selected_concepts(
  cid varchar(18) not null
  ,effectivetime timestamp not null
  ,PRIMARY KEY(cid)
);
create index concurrently active_selected_concepts_cid_ind on active_selected_concepts(cid);


drop table if exists active_descriptions cascade;
create table active_descriptions (
	did varchar(18) not null
	,effectivetime timestamp not null
	,term varchar(400) not null
	,cid varchar(18) not null
	,typeid varchar(18) not null
	,PRIMARY KEY(did)
	);
create index concurrently active_descriptions_did_ind on active_descriptions(did);


drop table if exists active_selected_descriptions cascade;
create table active_selected_descriptions (
	did varchar(18) not null
	,cid varchar(18) not null
	,term varchar(400) not null
	,effectivetime timestamp not null
	,PRIMARY KEY (did)
);
create index concurrently active_selected_descriptions_did_ind on active_selected_descriptions(did);