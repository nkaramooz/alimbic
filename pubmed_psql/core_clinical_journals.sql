set schema 'pubmed';

drop table if exists core_clinical_journals;
create table core_clinical_journals (
  iso_abbrev varchar(40)
  ,issn varchar(40)
  ,type varchar(40)
);

INSERT INTO core_clinical_journals(iso_abbrev, issn, type)
	VALUES
	('Acad Med', '1040-2446', 'Print'),
	('Acad Med', '1938-808X', 'Electronic'),
	('AJR Am J Roentgenol', '0361-803X', 'Print'),
	('AJR Am J Roentgenol', '1546-3141', 'Electronic'),
	('Am Fam Physician', '0002-838X', 'Print'),
	('Am Fam Physician', '1532-0650', 'Electronic'),
	('Am. Heart J.', '0002-8703', 'Print'),
	('Am. Heart J.', '1097-6744', 'Electronic'),
	('Am. J. Cardiol.', '0002-9149', 'Print'),
	('Am. J. Cardiol.', '1879-1913', 'Electronic'),
	('Am. J. Clin. Nutr.', '0002-9165', 'Print'),
	('Am. J. Clin. Nutr.', '1938-3207', 'Electronic'),
	('Am. J. Clin. Pathol.', '0002-9173', 'Print'),
	('Am. J. Clin. Pathol.', '1943-7722', 'Electronic'),
	('Am. J. Med.', '0002-9343', 'Print'),
	('Am. J. Med.', '1555-7162', 'Electronic'),
	('Am J Nurs', '0002-936X', 'Print'),
	('Am J Nurs', '1538-7488', 'Electronic'),
	('Am. J. Obstet. Gynecol.', '0002-9378', 'Print'),
	('Am. J. Obstet. Gynecol.', '1097-6868', 'Electronic'),
	('Am. J. Ophthalmol.', '0002-9394', 'Print'),
	('Am. J. Ophthalmol.', '1879-1891', 'Electronic'),
	('Am. J. Pathol.', '0002-9440', 'Print'),
	('Am. J. Pathol.', '1525-2191', 'Electronic'),
	('Am J Phys Med Rehabil', '0894-9115', 'Print'),
	('Am J Phys Med Rehabil', '1537-7385', 'Electronic'),
	('Am J Psychiatry', '0002-953X', 'Print'),
	('Am J Psychiatry', '1535-7228', 'Electronic'),
	('Am J Public Health', '0090-0036', 'Print'),
	('Am J Public Health', '1541-0048', 'Electronic'),
	('Am. J. Respir. Crit. Care Med.', '1073-449X', 'Print'),
	('Am. J. Respir. Crit. Care Med.', '1535-4970', 'Electronic'),
	('Am. J. Surg.', '0002-9610', 'Print'),
	('Am. J. Surg.', '1879-1883', 'Electronic'),
	('Am. J. Med. Sci.', '0002-9629', 'Print'),
	('Am. J. Med. Sci.', '1538-2990', 'Electronic'),
	('Am. J. Trop. Med. Hyg.', '0002-9637', 'Print'),
	('Am. J. Trop. Med. Hyg.', '1476-1645', 'Electronic'),
	('Anaesthesia', '0003-2409', 'Print'),
	('Anaesthesia', '1365-2044', 'Electronic'),
	('Anesth. Analg.', '0003-2999', 'Print'),
	('Anesth. Analg.', '1526-7598', 'Electronic'),
	('Anesthesiology', '0003-3022', 'Print'),
	('Anesthesiology', '1528-1175', 'Electronic'),
	('Ann Emerg Med', '0196-0644', 'Print'),
	('Ann Emerg Med', '1097-6760', 'Electronic'),
	('Ann. Intern. Med.', '0003-4819', 'Print'),
	('Ann. Intern. Med.', '1539-3704', 'Electronic'),
	('Ann. Otol. Rhinol. Laryngol.', '0003-4894', 'Print'),
	('Ann. Otol. Rhinol. Laryngol.', '1943-572X', 'Electronic'),
	('Ann. Surg.', '0003-4932', 'Print'),
	('Ann. Surg.', '1528-1140', 'Electronic'),
	('Ann. Thorac. Surg.', '0003-4975', 'Print'),
	('Ann. Thorac. Surg.', '1552-6259', 'Electronic'),
	('Arch. Dis. Child.', '0003-9888', 'Print'),
	('Arch. Dis. Child.', '1468-2044', 'Electronic'),
	('Arch. Dis. Child. Fetal Neonatal Ed.', '1359-2998', 'Print'),
	('Arch. Dis. Child. Fetal Neonatal Ed.', '1468-2052', 'Electronic'),
	('Arch Environ Occup Health', '1933-8244', 'Print'),
	('Arch. Pathol. Lab. Med.', '0003-9985', 'Print'),
	('Arch. Pathol. Lab. Med.', '1543-2165', 'Electronic'),
	('Arch Phys Med Rehabil', '0003-9993', 'Print'),
	('Arch Phys Med Rehabil', '1532-821X', 'Electronic'),
	('Arthritis Rheumatol', '2326-5191', 'Print'),
	('Arthritis Rheumatol', '2326-5205', 'Electronic'),
	('BJOG', '1470-0328', 'Print'),
	('BJOG', '1471-0528', 'Electronic'),
	('Blood', '0006-4971', 'Print'),
	('Blood', '1528-0020', 'Electronic'),
	('BMJ', '0959-8138', 'Print'),
	('BMJ', '1756-1833', 'Electronic'),
	('Bone Joint J', '2049-4394', 'Print'),
	('Bone Joint J', '2049-4408', 'Electronic'),
	('Brain', '0006-8950', 'Print'),
	('Brain', '1460-2156', 'Electronic'),
	('Br J Radiol', '0007-1285', 'Print'),
	('Br J Radiol', '1748-880X', 'Electronic'),
	('Br J Surg', '0007-1323', 'Print'),
	('Br J Surg', '1365-2168', 'Electronic'),
	('CA Cancer J Clin', '0007-9235', 'Print'),
	('CA Cancer J Clin', '1542-4863', 'Electronic'),
	('Cancer', '0008-543X', 'Print'),
	('Cancer', '1097-0142', 'Electronic'),
	('Chest', '0012-3692', 'Print'),
	('Chest', '1931-3543', 'Electronic'),
	('Circulation', '0009-7322', 'Print'),
	('Circulation', '0009-7322', 'Electronic'),
	('Clin. Orthop. Relat. Res.', '0009-921X', 'Print'),
	('Clin. Orthop. Relat. Res.', '1528-1132', 'Electronic'),
	('Clin Pediatr (Phila)', '0009-9228', 'Print'),
	('Clin Pediatr (Phila)', '1938-2707', 'Electronic'),
	('Clin. Pharmacol. Ther.', '0009-9236', 'Print'),
	('Clin. Pharmacol. Ther.', '1532-6535', 'Electronic'),
	('Clin Toxicol (Phila)', '1556-3650', 'Print'),
	('Clin Toxicol (Phila)', '1556-9519', 'Electronic'),
	('CMAJ', '0820-3946', 'Print'),
	('CMAJ', '1488-2329', 'Electronic'),
	('Crit. Care Med.', '0090-3493', 'Print'),
	('Crit. Care Med.', '1530-0293', 'Electronic'),
	('Curr Probl Surg', '0011-3840', 'Print'),
	('Curr Probl Surg', '1535-6337', 'Electronic'),
	('Diabetes', '0012-1797', 'Print'),
	('Diabetes', '1939-327X', 'Electronic'),
	('Dig. Dis. Sci.', '0163-2116', 'Print'),
	('Dig. Dis. Sci.', '1573-2568', 'Electronic'),
	('Dis Mon', '0011-5029', 'Print'),
	('Dis Mon', '1557-8194', 'Electronic'),
	('Endocrinology', '0013-7227', 'Print'),
	('Endocrinology', '1945-7170', 'Electronic'),
	('Gastroenterology', '0016-5085', 'Print'),
	('Gastroenterology', '1528-0012', 'Electronic'),
	('Gut', '0017-5749', 'Print'),
	('Gut', '1468-3288', 'Electronic'),
	('Heart', '1355-6037', 'Print'),
	('Heart', '1468-201X', 'Electronic'),
	('Heart Lung', '0147-9563', 'Print'),
	('Heart Lung', '1527-3288', 'Electronic'),
	('Hosp Pract (1995)', '2154-8331', 'Print'),
	('JAMA', '0098-7484', 'Print'),
	('JAMA', '1538-3598', 'Electronic'),
	('JAMA Dermatol', '2168-6068', 'Print'),
	('JAMA Dermatol', '2168-6084', 'Electronic'),
	('JAMA Intern Med', '2168-6106', 'Print'),
	('JAMA Intern Med', '2168-6114', 'Electronic'),
	('JAMA Neurol', '2168-6149', 'Print'),
	('JAMA Neurol', '2168-6157', 'Electronic'),
	('JAMA Ophthalmol', '2168-6165', 'Print'),
	('JAMA Ophthalmol', '2168-6173', 'Electronic'),
	('JAMA Otolaryngol Head Neck Surg', '2168-6181', 'Print'),
	('JAMA Otolaryngol Head Neck Surg', '2168-619X', 'Electronic'),
	('JAMA Pediatr', '2168-6203', 'Print'),
	('JAMA Pediatr', '2168-6211', 'Electronic'),
	('JAMA Psychiatry', '2168-622X', 'Print'),
	('JAMA Psychiatry', '2168-6238', 'Electronic'),
	('JAMA Surg', '2168-6254', 'Print'),
	('JAMA Surg', '2168-6262', 'Electronic'),
	('J. Allergy Clin. Immunol.', '0091-6749', 'Print'),
	('J. Allergy Clin. Immunol.', '1097-6825', 'Electronic'),
	('J Bone Joint Surg Am', '0021-9355', 'Print'),
	('J Bone Joint Surg Am', '1535-1386', 'Electronic'),
	('J Bone Joint Surg Am', '1058-2436', 'Undetermined'),
	('J. Clin. Endocrinol. Metab.', '0021-972X', 'Print'),
	('J. Clin. Endocrinol. Metab.', '1945-7197', 'Electronic'),
	('J. Clin. Invest.', '0021-9738', 'Print'),
	('J. Clin. Invest.', '1558-8238', 'Electronic'),
	('J. Clin. Pathol.', '0021-9746', 'Print'),
	('J. Clin. Pathol.', '1472-4146', 'Electronic'),
	('J Fam Pract', '0094-3509', 'Print'),
	('J Fam Pract', '1533-7294', 'Electronic'),
	('J. Immunol.', '0022-1767', 'Print'),
	('J. Immunol.', '1550-6606', 'Electronic'),
	('J. Infect. Dis.', '0022-1899', 'Print'),
	('J. Infect. Dis.', '1537-6613', 'Electronic'),
	('J Laryngol Otol', '0022-2151', 'Print'),
	('J Laryngol Otol', '1748-5460', 'Electronic'),
	('J. Nerv. Ment. Dis.', '0022-3018', 'Print'),
	('J. Nerv. Ment. Dis.', '1539-736X', 'Electronic'),
	('J. Neurosurg.', '0022-3085', 'Print'),
	('J. Neurosurg.', '1933-0693', 'Electronic'),
	('J Nurs Adm', '0002-0443', 'Print'),
	('J Nurs Adm', '1539-0721', 'Electronic'),
	('J. Oral Maxillofac. Surg.', '0278-2391', 'Print'),
	('J. Oral Maxillofac. Surg.', '1531-5053', 'Electronic'),
	('J. Pediatr.', '0022-3476', 'Print'),
	('J. Pediatr.', '1097-6833', 'Electronic'),
	('J Acad Nutr Diet', '2212-2672', 'Print'),
	('J. Am. Coll. Cardiol.', '0735-1097', 'Print'),
	('J. Am. Coll. Cardiol.', '1558-3597', 'Electronic'),
	('J. Am. Coll. Surg.', '1072-7515', 'Print'),
	('J. Am. Coll. Surg.', '1879-1190', 'Electronic'),
	('J. Thorac. Cardiovasc. Surg.', '0022-5223', 'Print'),
	('J. Thorac. Cardiovasc. Surg.', '1097-685X', 'Electronic'),
	('J Trauma Acute Care Surg', '2163-0755', 'Print'),
	('J Trauma Acute Care Surg', '2163-0763', 'Electronic'),
	('J. Urol.', '0022-5347', 'Print'),
	('J. Urol.', '1527-3792', 'Electronic'),
	('J. Gerontol. A Biol. Sci. Med. Sci.', '1079-5006', 'Print'),
	('J. Gerontol. A Biol. Sci. Med. Sci.', '1758-535X', 'Electronic'),
	('J Gerontol B Psychol Sci Soc Sci', '1079-5014', 'Print'),
	('J Gerontol B Psychol Sci Soc Sci', '1758-5368', 'Electronic'),
	('Lancet', '0140-6736', 'Print'),
	('Lancet', '1474-547X', 'Electronic'),
	('Mayo Clin. Proc.', '0025-6196', 'Print'),
	('Mayo Clin. Proc.', '1942-5546', 'Electronic'),
	('Med. Clin. North Am.', '0025-7125', 'Print'),
	('Med. Clin. North Am.', '1557-9859', 'Electronic'),
	('Med Lett Drugs Ther', '0025-732X', 'Print'),
	('Med Lett Drugs Ther', '1523-2859', 'Electronic'),
	('Medicine (Baltimore)', '0025-7974', 'Print'),
	('Medicine (Baltimore)', '1536-5964', 'Electronic'),
	('Neurology', '0028-3878', 'Print'),
	('Neurology', '1526-632X', 'Electronic'),
	('N. Engl. J. Med.', '0028-4793', 'Print'),
	('N. Engl. J. Med.', '1533-4406', 'Electronic'),
	('Nurs. Clin. North Am.', '0029-6465', 'Print'),
	('Nurs. Clin. North Am.', '1558-1357', 'Electronic'),
	('Nurs Outlook', '0029-6554', 'Print'),
	('Nurs Outlook', '1528-3968', 'Electronic'),
	('Nurs Res', '0029-6562', 'Print'),
	('Nurs Res', '1538-9847', 'Electronic'),
	('Obstet Gynecol', '0029-7844', 'Print'),
	('Obstet Gynecol', '1873-233X', 'Electronic'),
	('Orthop. Clin. North Am.', '0030-5898', 'Print'),
	('Orthop. Clin. North Am.', '1558-1373', 'Electronic'),
	('Pediatr. Clin. North Am.', '0031-3955', 'Print'),
	('Pediatr. Clin. North Am.', '1557-8240', 'Electronic'),
	('Pediatrics', '0031-4005', 'Print'),
	('Pediatrics', '1098-4275', 'Electronic'),
	('Phys Ther', '0031-9023', 'Print'),
	('Phys Ther', '1538-6724', 'Electronic'),
	('Plast. Reconstr. Surg.', '0032-1052', 'Print'),
	('Plast. Reconstr. Surg.', '1529-4242', 'Electronic'),
	('Postgrad Med', '0032-5481', 'Print'),
	('Postgrad Med', '1941-9260', 'Electronic'),
	('Prog Cardiovasc Dis', '0033-0620', 'Print'),
	('Prog Cardiovasc Dis', '1873-1740', 'Electronic'),
	('Public Health Rep', '0033-3549', 'Print'),
	('Public Health Rep', '1468-2877', 'Electronic'),
	('Radiol. Clin. North Am.', '0033-8389', 'Print'),
	('Radiol. Clin. North Am.', '1557-8275', 'Electronic'),
	('Radiology', '0033-8419', 'Print'),
	('Radiology', '1527-1315', 'Electronic'),
	('Rheumatology (Oxford)', '1462-0324', 'Print'),
	('Rheumatology (Oxford)', '1462-0332', 'Electronic'),
	('South. Med. J.', '0038-4348', 'Print'),
	('South. Med. J.', '1541-8243', 'Electronic'),
	('Surgery', '0039-6060', 'Print'),
	('Surgery', '1532-7361', 'Electronic'),
	('Surg. Clin. North Am.', '0039-6109', 'Print'),
	('Surg. Clin. North Am.', '1558-3171', 'Electronic'),
	('Transl Res', '1931-5244', 'Print'),
	('Transl Res', '1878-1810', 'Electronic'),
	('Urol. Clin. North Am.', '0094-0143', 'Print'),
	('Urol. Clin. North Am.', '1558-318X', 'Electronic')
	;

create index core_clin_issn_ind on core_clinical_journals(issn);
create index core_clin_iso_abbrev_ind on core_clinical_journals(iso_abbrev);