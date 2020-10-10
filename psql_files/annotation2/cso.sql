set schema 'annotation2';
drop table if exists cso;

create table cso as (
	select tb2.id, tb2.conceptid 
	from annotation.sentences5 tb2
	inner join annotation.concept_types tb3
		on tb2.conceptid = tb3.root_cid and rel_type in ('condition', 'symptom', 'organism')
);

create index cso_id_ind on cso(id);
create index cso_conceptid_ind on cso(conceptid);