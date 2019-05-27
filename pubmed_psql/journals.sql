set schema 'pubmed';

drop table if exists journals;
create table journals as (
	select iso_abbrev, issn, type
	from pubmed.core_clinical_journals

	union

	select iso_abbrev, issn, type
	from pubmed.additional_journals
);


create index journals_issn_ind on journals(issn);
create index journals_iso_abbrev_ind on journals(iso_abbrev);