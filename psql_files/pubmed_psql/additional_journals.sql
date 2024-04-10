set schema 'pubmed';

drop table if exists additional_journals;
create table additional_journals (
  iso_abbrev varchar(40)
  ,issn varchar(40) 
  ,type varchar(40)
);

INSERT INTO additional_journals(iso_abbrev, issn, type)
	VALUES
	('Test Journal', '9999-9999', 'Eletronic'),
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

	('Respir Med Case Rep', '2213-0071', 'Electronic'),
	('Clin Med (Lond)', '1473-4893', 'Print'),
	('Clin Med (Lond)', '1470-2118', 'Electronic'),
	('Pediatr Emerg Care', '0749-5161', 'Print'),
	('Pediatr Emerg Care', '1535-1815', 'Electronic'),
	('Cardiovasc Intervent Radiol', '0174-1551', 'Print'),
	('Cardiovasc Intervent Radiol', '1432-086X', 'Electronic'),
	('J Vasc Interv Radiol', '1051-0443', 'Print'),
	('J Vasc Interv Radiol', '1535-7732', 'Electronic'),
	('Trends Cardiovasc Med', '1050-1738', 'Print'),
	('Trends Cardiovasc Med', '1873-2615', 'Electronic'),
	('Arch Cardiovasc Dis', '1875-2136', 'Print'),
	('Arch Cardiovasc Dis', '1875-2128', 'Electronic'),
	('Br J Cancer', '0007-0920', 'Print'),
	('Br J Cancer', '1532-1827', 'Electronic'),
	('Eur J Cancer', '0959-8049', 'Print'),
	('Eur J Cancer', '1879-0852', 'Electronic'),
	('Clin J Gastroenterol', '1865-7257', 'Print'),
	('Clin J Gastroenterol', '1865-7265', 'Electronic'),
	('Clin J Gastroenterol', '1865-7257', 'Print'),
	('Clin J Gastroenterol', '1865-7265', 'Electronic'),
	('Clin Gastroenterol Hepatol', '1542-3565', 'Print'),
	('Clin Gastroenterol Hepatol', '1542-7714', 'Electronic'),
	('Am J Gastroenterol', '0002-9270', 'Print'),
	('Am J Gastroenterol', '1572-0241', 'Electronic'),
	('World J Surg Oncol', '1477-7819', 'Electronic'),
	('Australas Radiol', '0004-8461', 'Print'),
	('Australas Radiol', '1440-1673', 'Electronic'),
	('Rev Urol', '1523-6161', 'Print'),
	('Rev Urol', '2153-8182', 'Electronic'),
	('Nat Clin Pract Urol', '1743-4270', 'Print'),
	('Nat Clin Pract Urol', '1743-4289', 'Electronic'),
	('Indian J Orthop', '0019-5413', 'Print'),
	('Indian J Orthop', '1998-3727', 'Electronic'),
	('Am J Manag Care', '1088-0224', 'Print'),
	('Am J Manag Care', '1936-2692', 'Electronic'),
	('Prim Care', '0095-4543', 'Print'),
	('Prim Care', '1558-299X', 'Electronic'),
	('Pediatr Rev', '0191-9601', 'Print'),
	('Pediatr Rev', '1526-3347', 'Electronic'),
	('PLoS One', '1932-6203', 'Electronic'),
	('Circ Arrhythm Electrophysiol', '1941-3149', 'Print'),
	('Circ Arrhythm Electrophysiol', '1941-3084', 'Electronic'),
	('Urol Case Rep', '2214-4420', 'Undetermined'),
	('J Med Case Rep', '1752-1947', 'Undetermined'),
	('J Antimicrob Chemother', '0305-7453', 'Print'),
	('J Antimicrob Chemother', '1460-2091', 'Electronic'),
	('J Am Soc Nephrol', '1046-6673', 'Print'),
	('J Am Soc Nephrol', '1533-3450', 'Electronic'),
	('Am J Hemat', '0361-8609', 'Print'),
	('Am J Hemat', '1096-8652', 'Electronic'),
	('Heart Fail Rev', '1382-4147', 'Print'),
	('Heart Fail Rev', '1573-7322', 'Electronic'),
	('Eur J Heart Fail', '1388-9842', 'Print'),
	('Eur J Heart Fail', '1879-0844', 'Electronic'),
	('Oncologist', '1083-7159', 'Print'),
	('Oncologist', '1549-490X', 'Electronic'),
	('Int J Cardiol', '0167-5273', 'Print'),
	('Int J Cardiol', '1874-1754', 'Electronic'),
	('Respir Med', '0954-6111', 'Print'),
	('Respir Med', '1532-3064', 'Electronic'),
	('BMJ Open Respir Res', '2052-4439', 'Electronic'),
	
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
	('Curr Opin Neurol', '1473-6551', 'Electronic'),
	('Int J Mol Sci', '1422-0067', 'Electronic'),
	('Leuk Lymphoma', '1042-8194', 'Print'),
	('Leuk Lymphoma', '1029-2403', 'Electronic'),
	('Eur J Drug Metab Pharmacokinet', '0378-7966', 'Print'),
	('Eur J Drug Metab Pharmacokinet', '2107-0180', 'Electronic'),
	('Cancer Chemother Pharmacol', '0344-5704', 'Print'),
	('Cancer Chemother Pharmacol', '1432-0843', 'Electronic'),
	('JAMA Oncol', '2374-2437', 'Print'),
	('JAMA Oncol', '2374-2445', 'Electronic'),
	('Int Immunopharmacol', '1567-5769', 'Print'),
	('Int Immunopharmacol', '1878-1705', 'Electronic'),
	('Eur J Med Chem', '0223-5234', 'Print'),
	('Eur J Med Chem', '1768-3254', 'Electronic'),
	('Curr Med Chem', '0929-8673', 'Print'),
	('Curr Med Chem', '1875-533X', 'Electronic'),
	('Ann Hematol', '0939-5555', 'Print'),
	('Ann Hematol', '1432-0584', 'Electronic'),
	('Transfusion', '0041-1132', 'Print'),
	('Transfusion', '1537-2995', 'Electronic'),
	('Br J Haematol', '0007-1048', 'Print'),
	('Br J Haematol', '1365-2141', 'Electronic'),
	('Curr Rheumatol Rev', '1573-3971', 'Print'),
	('Curr Rheumatol Rev', '1875-6360', 'Electronic'),
	('J Clin Oncol', '0732-183X', 'Print'),
	('J Clin Oncol', '1527-7755', 'Electronic'),
	('J Clin Oncol', '0732-183X', 'Print'),
	('J Clin Oncol', '1527-7755', 'Electronic'),
	('Hypertension', '0194-911X', 'Print'),
	('Hypertension', '1524-4563', 'Electronic'),
	('Acad Emerg Med', '1069-6563', 'Print'),
	('Acad Emerg Med', '1553-2712', 'Electronic'),
	('Nature', '0028-0836', 'Print'),
	('Nature', '1476-4687', 'Electronic'),
	('Clin Infect Dis', '1058-4838', 'Print'),
	('Clin Infect Dis', '1537-6591', 'Electronic'),
	('Clin Cardiol', '0160-9289', 'Print'),
	('Clin Cardiol', '1932-8737', 'Electronic'),
	('Clin Res Cardiol', '1861-0684', 'Print'),
	('Clin Res Cardiol', '1861-0692', 'Electronic'),
	('Cardiol Rev', '1061-5377', 'Print'),
	('Cardiol Rev', '1538-4683', 'Electronic'),
	('J Electrocardiol', '0022-0736', 'Print'),
	('J Electrocardiol', '1532-8430', 'Electronic'),
	('Nat Rev Nephrol', '1759-5061', 'Print'),
	('Nat Rev Nephrol', '1759-507X', 'Electronic'),
	('Kidney Int Rep', '2468-0249', 'Electronic'),
	('J Gen Intern Med', '0884-8734', 'Print'),
	('J Gen Intern Med', '1525-1497', 'Electronic'),
	('Int J Gen Med', '1178-7074', 'Electronic'),
	('J Hosp Med', '1553-5592', 'Print'),
	('J Hosp Med', '1553-5606', 'Electronic'),
	('Endocrine', '1355-008X', 'Print'),
	('Endocrine', '1559-0100', 'Electronic'),
	('Endocr J', '0918-8959', 'Print'),
	('Endocr J', '1348-4540', 'Electronic'),
	('Endocr Rev', '0163-769X', 'Print'),
	('Endocr Rev', '1945-7189', 'Electronic'),
	('JACC Clin Electrophysiol', '2405-500X', 'Print'),
	('JACC Clin Electrophysiol', '2405-5018', 'Electronic'),
	('J Rheumatol', '0315-162X', 'Print'),
	('Clin Rheumatol', '0770-3198', 'Print'),
	('Clin Rheumatol', '1434-9949', 'Electronic'),
	('Arthritis Rheum', '0004-3591', 'Print'),
	('Arthritis Rheum', '1529-0131', 'Electronic'),
	('J Clin Rheumatol', '1076-1608', 'Print'),
	('J Clin Rheumatol', '1536-7355', 'Electronic'),
	('Ann Rheum Dis', '0003-4967', 'Print'),
	('Ann Rheum Dis', '1468-2060', 'Electronic'),
	('Best Pract Res Clin Rheumatol', '1521-6942', 'Print'),
	('Best Pract Res Clin Rheumatol', '1532-1770', 'Electronic'),
	('Nat Rev Rheumatol', '1759-4790', 'Print'),
	('Nat Rev Rheumatol', '1759-4804', 'Electronic'),
	('Curr Rheumatol Rep', '1523-3774', 'Print'),
	('Curr Rheumatol Rep', '1534-6307', 'Electronic'),
	('J Clin Gastroenterol', '0192-0790', 'Print'),
	('J Clin Gastroenterol', '1539-2031', 'Electronic'),
	('Allergy Asthma Proc', '1088-5412', 'Print'),
	('Allergy Asthma Proc', '1539-6304', 'Electronic'),
	('Ther Clin Risk Manag', '1176-6336', 'Print'),
	('Ther Clin Risk Manag', '1178-203X', 'Electronic'),
	('Int Arch Allergy Immunol', '1018-2438', 'Print'),
	('Int Arch Allergy Immunol', '1423-0097', 'Electronic'),
	('Drugs', '0012-6667', 'Print'),
	('Drugs', '1179-1950', 'Electronic'),
	('J Am Acad Orthop Surg', '1067-151X', 'Print'),
	('J Am Acad Orthop Surg', '1940-5480', 'Electronic'),
	('Phys Ther Sport', '1466-853X', 'Print'),
	('Phys Ther Sport', '1873-1600', 'Electronic'),
	('Del Med J', '0011-7781', 'Print'),
	('BMJ Case Rep', '1757-790X', 'Electronic')
	;

create index add_journals_issn_ind on additional_journals(issn);
create index add_journals_iso_abbrev_ind on additional_journals(iso_abbrev);