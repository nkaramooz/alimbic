set schema 'annotation2';
	
drop table if exists annotation2.new_concepts;
create table new_concepts(
	cid varchar(36) not null
	,did varchar(36) not null
	,effectivetime timestamp not null
	,PRIMARY KEY (cid)
);
create index concurrently new_concepts_cid_ind on new_concepts(cid);
create index concurrently new_concepts_did_ind on new_concepts(did);


insert into new_concepts
	select 
		conceptid as cid
		,description_id as did
		,effectivetime
	from annotation.new_concepts
;

drop table if exists manual_active_desc;
create table manual_active_desc(
	cid varchar(36) not null
	,did varchar(36) not null
	,term varchar(400) not null
	,effectivetime timestamp
	,PRIMARY KEY (did)
);
create index concurrently manual_active_desc_cid_ind on manual_active_desc(cid);
create index concurrently manual_active_desc_did_ind on manual_active_desc(did);
create index concurrently manual_active_desc_term_ind on manual_active_desc(term);


insert into manual_active_desc
	select 
		conceptid as cid
		,description_id as did
		,t1.term as term
		,now() as effectivetime
	from annotation.description_whitelist t1
	left outer join annotation2.manual_inactive_desc t2
	on t1.conceptid = t2.cid and t1.term = t2.term
;

insert into manual_active_desc
	select
		conceptid as cid
		,description_id as did
		,description as term
		,effectivetime
	from annotation.new_concepts
;