set schema 'annotation2';

drop table if exists acronym_override;
create table acronym_override(
	id varchar(36) not null
	,did varchar(36) not null
	,is_acronym boolean not null
	,effectivetime timestamp not null
	,PRIMARY KEY (id)
);
create index concurrently acronym_override_did_ind on acronym_override(did);


insert into acronym_override 
	select
		t1.id
		,t1.description_id as did
		,case when is_acronym=0 then 'f' else 't' end ::boolean as is_acronym
		,t1.effectivetime
	from annotation.acronym_override t1
;