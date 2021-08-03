set schema 'annotation';

drop table if exists downstream_root_cid cascade;
create table downstream_root_cid(
	acid varchar(36) not null
	,cid varchar(36) not null
	,active boolean not null
	,effectivetime timestamp not null
	,unique(cid)
);
create index concurrently downstream_root_cid_acid_ind on downstream_root_cid(acid);
create index concurrently downstream_root_cid_cid_ind on downstream_root_cid(cid);


drop table if exists downstream_root_did cascade;
create table downstream_root_did(
	adid varchar(36) not null
	,did varchar(36) not null
	,acid varchar(36) not null
	,term varchar(400) not null
	,active boolean not null
	,effectivetime timestamp not null
	,unique(did, term)
);
create index concurrently downstream_root_did_adid_ind on downstream_root_did(adid);
create index concurrently downstream_root_did_did_ind on downstream_root_did(did);	

insert into downstream_root_cid
	select 
		acid::varchar(36) as acid
		,cid 
		,active
		,effectivetime
	from (
		select 
			t1.acid
			,t1.cid
			,case when t1.effectivetime < t3.effectivetime then t3.active else t1.active end as active
			,case when t1.effectivetime < t3.effectivetime then t3.effectivetime else t1.effectivetime end as effectivetime
		from annotation.upstream_root_cid t1
		left join (
			select 
				acid
				,active
				,effectivetime
			from (
				select 
					acid
					,active
					,effectivetime
					,row_number() over (partition by acid order by effectivetime desc) as row_num
				from annotation.inactive_concepts
			) t2
			where row_num = 1
		) t3
		on t1.acid = t3.acid
	) t4
	where active='t'
	ON CONFLICT(cid) DO NOTHING
;

insert into downstream_root_did
	select 
		adid::varchar(36)
		,did
		,acid::varchar(36)
		,term
		,active
		,effectivetime
	from (
		select 
			t1.adid
			,t1.did
			,t1.acid
			,t1.term
			,case when t1.effectivetime < t3.effectivetime then t3.active else t1.active end as active
			,case when t1.effectivetime < t3.effectivetime then t3.effectivetime else t1.effectivetime end as effectivetime
		from annotation.upstream_root_did t1
		left join (
			select 
				adid
				,active
				,effectivetime
			from (
				select 
					adid
					,active
					,effectivetime
					,row_number () over (partition by adid order by effectivetime desc) as row_num
				from annotation.root_desc_inactive
				) t2
			where row_num=1
		) t3
		on t1.adid::varchar(36) = t3.adid
	) t4
	where active='t' and acid::varchar(36) not in (select acid from annotation.downstream_root_cid where active='f')
	ON CONFLICT(did, term) DO NOTHING
;
