set schema 'pubmed';

drop table if exists additional_journals;
create table additional_journals (
  iso_abbrev varchar(40)
  ,issn varchar(40)
  ,type varchar(40)
);

INSERT INTO additional_journals(iso_abbrev, issn, type)
	VALUES
	('Cochrane Database Syst Rev', '1469-493X', 'Electronic'),
	('BMJ Case Rep', '1757-790X', 'Electronic'),
	('Hypertension', '0194-911X', 'Print'),
	('Hypertension', '1524-4563', 'Electronic'),
	('JACC Heart Fail', '2213-1779', 'Print'),
	('JACC Heart Fail', '2213-1787', 'Electronic')
	;

create index add_journals_issn_ind on additional_journals(issn);
create index add_journals_iso_abbrev_ind on additional_journals(iso_abbrev);