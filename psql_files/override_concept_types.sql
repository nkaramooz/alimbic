set schema 'annotation';
create table override_concept_types(
  root_cid varchar(40)
  ,associated_cid varchar(40)
  ,rel_type text
  ,active integer
  ,effectivetime timestamp
);


create index o_ct_conceptid on override_concept_types(root_cid);
create index o_ct_concept_type on override_concept_types(rel_type)