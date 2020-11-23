set schema 'ml2';

drop table if exists labelled_treatments_seed;
create table labelled_treatments_seed (
  condition_acid varchar(40)
  ,treatment_acid varchar(40)
  ,label integer
);

-- value of 2 = too broad or partial treatment term
-- will not be displayed in app

INSERT INTO ml2.labelled_treatments_seed(condition_acid, treatment_acid, label)
	VALUES
	('154355', '336184', 0)
	,('112031', '38869', 0)
	,('112031', '326523', 0)
	,('%', '428053', 2)
	,('112031', '507464', 0)
	,('%', '69109', 2)
	,('112031', '436846', 0)
	,('138688', '366023', 0)
	,('44839', '366023', 0)
	,('44839', '106489', 0)
	,('154355', '174980', 0)
	,('58782', '1709', 0)
	,('10609', '75715', 0)
	,('10609', '260685', 0)
	,('10609', '766240', 0)
	,('10609', '6190', 0)
	,('%', '21416', 0)
	,('%', '213104', 0)
	,('%', '250848', 0)
	,('%', '132114', 0)
	;
