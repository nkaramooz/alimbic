set schema 'guidelines';

drop table if exists guideline_orgs;
create table guideline_orgs(
	org_full varchar(100) not null
	,org_acr varchar(40) not null
	,url varchar(400) not null
);
create index guideline_orgs_org_acr on guideline_orgs(org_acr);



INSERT INTO guidelines.guideline_orgs(org_full, org_acr, url)
	VALUES 
	('American College of Obstetricians and Gynecologists', 'ACOG', 'https://www.acog.org/clinical/clinical-guidance/practice-bulletin')
	,('American College of Cardiology', 'ACC', 'https://www.acc.org/guidelines')
	,('European Society of Cardiology', 'ESC', 'https://www.acc.org/guidelines')
	,('American College of Gastroenterology', 'ACG', 'https://gi.org/guidelines/')
	,('Infectious Disease Society of America', 'IDSA', 'https://www.idsociety.org/practice-guideline/alphabetical-guidelines/')
	,('American College of Rheumatology', 'ACR', 'https://www.rheumatology.org/Practice-Quality/Clinical-Support/Clinical-Practice-Guidelines')
	,('National Comprehensive Cancer Network', 'NCCN', 'https://www.nccn.org/guidelines/category_1')
;