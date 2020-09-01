set schema 'annotation2';


-- drop table if exists cleaned_selected_descriptions_prelim cascade;
-- create table cleaned_selected_descriptions_prelim(
--   did varchar(18) not null
--   ,cid varchar(18) not null
--   ,term varchar(400) not null
--   ,PRIMARY KEY(did)
-- );
-- create index concurrently cleaned_selected_descriptions_prelim_did_ind on cleaned_selected_descriptions_prelim(did);
-- create index concurrently cleaned_selected_descriptions_prelim_cid_ind on cleaned_selected_descriptions_prelim(cid);


drop table if exists cleaned_selected_descriptions_de_duped;
create table cleaned_selected_descriptions_de_duped (
	did varchar(18) not null
	,cid varchar(18) not null
	,term varchar(400) not null
	,effectivetime timestamp not null
	,PRIMARY KEY (did)
);
create index concurrently cleaned_selected_descriptions_de_duped_cid_ind on cleaned_selected_descriptions_de_duped(cid);
create index concurrently cleaned_selected_descriptions_de_duped_did_ind on cleaned_selected_descriptions_de_duped(did);


drop table if exists acronym_augmented_descriptions;
create table acronym_augmented_descriptions (
	cid varchar(18) not null
	,did varchar(18) not null
	,term varchar(400) not null
	,candidate varchar(400) not null
	,effectivetime timestamp not null
);
create index concurrently acronym_augmented_descriptions_cid_ind on acronym_augmented_descriptions(cid);
create index concurrently acronym_augmented_descriptions_candidate_ind on acronym_augmented_descriptions(candidate);


drop table if exists snomed_synonyms;
create table snomed_synonyms (
	ref_cid varchar(18) not null
	,ref_term varchar(400) not null
	,ref_rank integer not null
	,syn_cid varchar(18) not null
	,syn_rank integer not null
);
create index concurrently snomed_synonyms_ref_cid_ind on snomed_synonyms(ref_cid);
create index concurrently snomed_synonyms_syn_cid_ind on snomed_synonyms(syn_cid);


drop table if exists snomed_cid_ignore;
create table snomed_cid_ignore (
	cid varchar(18) not null
	,PRIMARY KEY (cid)
	,unique(cid)
);
create index concurrently snomed_cid_ignore_cid_ind on snomed_cid_ignore(cid);


drop table if exists therapies_synonyms;
create table therapies_synonyms(
	cid varchar(18) not null
	,did varchar(18) not null
	,term varchar(400) not null
	,effectivetime timestamp not null
);
create index concurrently therapies_syn_term_ind on therapies_synonyms(term);
create index concurrently therapies_syn_cid_ind on therapies_synonyms(cid);
create index concurrently therapies_syn_did_ind on therapies_synonyms(did);