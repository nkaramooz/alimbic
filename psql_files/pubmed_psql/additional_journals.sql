set schema 'pubmed';

-- drop table if exists additional_journals;
-- create table additional_journals (
--   iso_abbrev varchar(40)
--   ,issn varchar(40)
--   ,type varchar(40)
-- );

INSERT INTO additional_journals(iso_abbrev, issn, type)
	VALUES
	('ACG Case Rep J', '2326-3253', 'Electronic'),
	('Case Rep Cardiol', '2090-6404', 'Print'),
	('Case Rep Cardiol', '2090-6412', 'Print'),

	('Case Rep Dermatol Med', '2090-6463', 'Print'),
	('Case Rep Dermatol Med', '2090-6471', 'Electronic'),

	('Case Rep Dermatol', '1662-6567', 'Electronic'),

	('Case Rep Rheumatol', '2090-6889', 'Print'),
	('Case Rep Rheumatol', '2090-6897', 'Electronic'),

	('Clin Case Rep', '2050-0904', 'Electronic'),

	('Am J Med Case Rep', '2374-2151', 'Print'),
	('Am J Med Case Rep', '2374-216X', 'Electronic'),

	('Respir Med Case Rep', '2213-0071', 'Electronic')
	;

-- create index add_journals_issn_ind on additional_journals(issn);
-- create index add_journals_iso_abbrev_ind on additional_journals(iso_abbrev);