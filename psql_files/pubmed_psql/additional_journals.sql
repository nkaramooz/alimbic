set schema 'pubmed';

drop table if exists additional_journals;
create table additional_journals (
  iso_abbrev varchar(40)
  ,issn varchar(40) 
  ,type varchar(40)
);

INSERT INTO additional_journals(iso_abbrev, issn, type)
	VALUES
	-- ('ACG Case Rep J', '2326-3253', 'Electronic'),
	-- ('Case Rep Cardiol', '2090-6404', 'Print'),
	-- ('Case Rep Cardiol', '2090-6412', 'Print'),

	-- ('Case Rep Dermatol Med', '2090-6463', 'Print'),
	-- ('Case Rep Dermatol Med', '2090-6471', 'Electronic'),

	-- ('Case Rep Dermatol', '1662-6567', 'Electronic'),

	-- ('Case Rep Rheumatol', '2090-6889', 'Print'),
	-- ('Case Rep Rheumatol', '2090-6897', 'Electronic'),

	-- ('Clin Case Rep', '2050-0904', 'Electronic'),

	-- ('Am J Med Case Rep', '2374-2151', 'Print'),
	-- ('Am J Med Case Rep', '2374-216X', 'Electronic'),

	-- ('Respir Med Case Rep', '2213-0071', 'Electronic'),
	-- ('Clin Med (Lond)', '1473-4893', 'Print'),
	-- ('Clin Med (Lond)', '1470-2118', 'Electronic'),
	-- ('Pediatr Emerg Care', '0749-5161', 'Print'),
	-- ('Pediatr Emerg Care', '1535-1815', 'Electronic'),
	-- ('Cardiovasc Intervent Radiol', '0174-1551', 'Print'),
	-- ('Cardiovasc Intervent Radiol', '1432-086X', 'Electronic'),
	-- ('J Vasc Interv Radiol', '1051-0443', 'Print'),
	-- ('J Vasc Interv Radiol', '1535-7732', 'Electronic'),
	-- ('Trends Cardiovasc Med', '1050-1738', 'Print'),
	-- ('Trends Cardiovasc Med', '1873-2615', 'Electronic'),
	-- ('Arch Cardiovasc Dis', '1875-2136', 'Print'),
	-- ('Arch Cardiovasc Dis', '1875-2128', 'Electronic'),
	-- ('Br J Cancer', '0007-0920', 'Print'),
	-- ('Br J Cancer', '1532-1827', 'Electronic'),
	-- ('Eur J Cancer', '0959-8049', 'Print'),
	-- ('Eur J Cancer', '1879-0852', 'Electronic'),
	-- ('Clin J Gastroenterol', '1865-7257', 'Print'),
	-- ('Clin J Gastroenterol', '1865-7265', 'Electronic'),
	-- ('Clin J Gastroenterol', '1865-7257', 'Print'),
	-- ('Clin J Gastroenterol', '1865-7265', 'Electronic'),
	-- ('Clin Gastroenterol Hepatol', '1542-3565', 'Print'),
	-- ('Clin Gastroenterol Hepatol', '1542-7714', 'Electronic'),
	-- ('Am J Gastroenterol', '0002-9270', 'Print'),
	-- ('Am J Gastroenterol', '1572-0241', 'Electronic'),
	-- ('World J Surg Oncol', '1477-7819', 'Electronic'),
	-- ('Australas Radiol', '0004-8461', 'Print'),
	-- ('Australas Radiol', '1440-1673', 'Electronic'),
	-- ('Rev Urol', '1523-6161', 'Print'),
	-- ('Rev Urol', '2153-8182', 'Electronic'),
	-- ('Nat Clin Pract Urol', '1743-4270', 'Print'),
	-- ('Nat Clin Pract Urol', '1743-4289', 'Electronic'),
	-- ('Indian J Orthop', '0019-5413', 'Print'),
	-- ('Indian J Orthop', '1998-3727', 'Electronic'),
	-- ('Am J Manag Care', '1088-0224', 'Print'),
	-- ('Am J Manag Care', '1936-2692', 'Electronic'),
	-- ('Prim Care', '0095-4543', 'Print'),
	-- ('Prim Care', '1558-299X', 'Electronic'),
	-- ('Pediatr Rev', '0191-9601', 'Print'),
	-- ('Pediatr Rev', '1526-3347', 'Electronic'),
	-- ('PLoS One', '1932-6203', 'Electronic'),
	-- ('Circ Arrhythm Electrophysiol', '1941-3149', 'Print'),
	-- ('Circ Arrhythm Electrophysiol', '1941-3084', 'Electronic'),
	-- ('Urol Case Rep', '2214-4420', 'Undetermined'),
	-- ('J Med Case Rep', '1752-1947', 'Undetermined'),
	-- ('J Antimicrob Chemother', '0305-7453', 'Print'),
	-- ('J Antimicrob Chemother', '1460-2091', 'Electronic'),
	-- ('J Am Soc Nephrol', '1046-6673', 'Print'),
	-- ('J Am Soc Nephrol', '1533-3450', 'Electronic'),
	-- ('Am J Hemat', '0361-8609', 'Print'),
	-- ('Am J Hemat', '1096-8652', 'Electronic'),
	-- ('Heart Fail Rev', '1382-4147', 'Print'),
	-- ('Heart Fail Rev', '1573-7322', 'Electronic'),
	-- ('Eur J Heart Fail', '1388-9842', 'Print'),
	-- ('Eur J Heart Fail', '1879-0844', 'Electronic'),
	-- ('Oncologist', '1083-7159', 'Print'),
	-- ('Oncologist', '1549-490X', 'Electronic'),
	-- ('Int J Cardiol', '0167-5273', 'Print'),
	-- ('Int J Cardiol', '1874-1754', 'Electronic'),
	-- ('Respir Med', '0954-6111', 'Print'),
	-- ('Respir Med', '1532-3064', 'Electronic'),
	-- ('BMJ Open Respir Res', '2052-4439', 'Electronic'),
	
	('Eur Heart J', '0195-668X', 'Print'),
	('Eur Heart J', '1522-9645', 'Electronic'),
	('ESC Heart Fail', '2055-5822', 'Electronic'),
	('Eur J Haematol', '0902-4441', 'Print'),
	('Eur J Haematol', '1600-0609', 'Electronic'),
	('Heart Fail Clin', '1551-7136', 'Electronic'),
	('Curr Heart Fail Rep', '1546-9530', 'Print'),
	('Curr Heart Fail Rep', '1546-9549', 'Electronic'),
	('Eur J Paediatr Neurol', '1090-3798', 'Print'),
	('Eur J Paediatr Neurol', '1532-2130', 'Electronic'),
	('Phys Med Rehabil Clin N Am', '1047-9651', 'Print'),
	('Phys Med Rehabil Clin N Am', '1558-1381', 'Electronic'),
	('J Hand Surg Am', '0363-5023', 'Print'),
	('J Hand Surg Am', '1531-6564', 'Electronic'),
	('Muscle Nerve', '0148-639X', 'Print'),
	('Muscle Nerve', '1097-4598', 'Electronic'),
	('Joint Bone Spine', '1297-319X', 'Print'),
	('Joint Bone Spine', '1778-7254', 'Electronic'),
	('Mult Scler', '1352-4585', 'Print'),
	('Mult Scler', '1477-0970', 'Electronic'),
	('Neurol Clin', '0733-8619', 'Print'),
	('Neurol Clin', '1557-9875', 'Electronic'),
	('Curr Opin Neurol', '1350-7540', 'Print'),
	('Curr Opin Neurol', '1473-6551', 'Electronic')
	;

create index add_journals_issn_ind on additional_journals(issn);
create index add_journals_iso_abbrev_ind on additional_journals(iso_abbrev);