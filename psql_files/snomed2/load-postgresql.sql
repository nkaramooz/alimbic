/* loads the SNOMED CT 'Full' release - replace filenames with relevant locations of base SNOMED CT release files*/
/* Filenames may need to change depending on the release you wish to upload, currently set to January 2015 release */

set schema 'snomed2';

COPY curr_concept_f(id, effectivetime, active, moduleid, definitionstatusid)
FROM '/home/nkaramooz/Documents/alimbic/resources/snomed_files/Terminology/sct2_Concept_Snapshot_US1000124_20200901.txt' 
WITH (FORMAT csv, HEADER true, DELIMITER '	', QUOTE E'\b');

COPY curr_description_f(id, effectivetime, active, moduleid, conceptid, languagecode, typeid, term, casesignificanceid)
FROM '/home/nkaramooz/Documents/alimbic/resources/snomed_files/Terminology/sct2_Description_Snapshot-en_US1000124_20200901.txt'
WITH (FORMAT csv, HEADER true, DELIMITER '	', QUOTE E'\b');

COPY curr_textdefinition_f(id, effectivetime, active, moduleid, conceptid, languagecode, typeid, term, casesignificanceid)
FROM '/home/nkaramooz/Documents/alimbic/resources/snomed_files/Terminology/sct2_TextDefinition_Snapshot-en_US1000124_20200901.txt'
WITH (FORMAT csv, HEADER true, DELIMITER '	', QUOTE E'\b');

COPY curr_relationship_f(id, effectivetime, active, moduleid, sourceid, destinationid, relationshipgroup, typeid,characteristictypeid, modifierid)
FROM '/home/nkaramooz/Documents/alimbic/resources/snomed_files/Terminology/sct2_Relationship_Snapshot_US1000124_20200901.txt'
WITH (FORMAT csv, HEADER true, DELIMITER '	', QUOTE E'\b');

COPY curr_stated_relationship_f(id, effectivetime, active, moduleid, sourceid, destinationid, relationshipgroup, typeid,  characteristictypeid, modifierid)
FROM '/home/nkaramooz/Documents/alimbic/resources/snomed_files/Terminology/sct2_StatedRelationship_Snapshot_US1000124_20200901.txt'
WITH (FORMAT csv, HEADER true, DELIMITER '	', QUOTE E'\b');

COPY curr_langrefset_f(id, effectivetime, active, moduleid, refsetid, referencedcomponentid, acceptabilityid)
FROM '/home/nkaramooz/Documents/alimbic/resources/snomed_files/Refset/Language/der2_cRefset_LanguageSnapshot-en_US1000124_20200901.txt'
WITH (FORMAT csv, HEADER true, DELIMITER '	', QUOTE E'\b');

COPY curr_associationrefset_d(id, effectivetime, active, moduleid, refsetid, referencedcomponentid, targetcomponentid)
FROM '/home/nkaramooz/Documents/alimbic/resources/snomed_files/Refset/Content/der2_cRefset_AssociationSnapshot_US1000124_20200901.txt'
WITH (FORMAT csv, HEADER true, DELIMITER '	', QUOTE E'\b');

COPY curr_simplerefset_f(id, effectivetime, active, moduleid, refsetid, referencedcomponentid)
FROM '/home/nkaramooz/Documents/alimbic/resources/snomed_files/Refset/Content/der2_Refset_SimpleSnapshot_US1000124_20200901.txt'
WITH (FORMAT csv, HEADER true, DELIMITER '	', QUOTE E'\b');

COPY curr_attributevaluerefset_f(id, effectivetime, active, moduleid, refsetid, referencedcomponentid, valueid)
FROM '/home/nkaramooz/Documents/alimbic/resources/snomed_files/Refset/Content/der2_cRefset_AttributeValueSnapshot_US1000124_20200901.txt'
WITH (FORMAT csv, HEADER true, DELIMITER '	', QUOTE E'\b');

COPY curr_simplemaprefset_f(id, effectivetime, active, moduleid, refsetid,  referencedcomponentid, maptarget)
FROM '/home/nkaramooz/Documents/alimbic/resources/snomed_files/Refset/Map/der2_sRefset_SimpleMapSnapshot_US1000124_20200901.txt'
WITH (FORMAT csv, HEADER true, DELIMITER '	', QUOTE E'\b');

COPY curr_extendedmaprefset_f(id, effectivetime, active, moduleid, refsetid, referencedcomponentid, mapGroup, mapPriority, mapRule, mapAdvice, mapTarget, correlationId, mapCategoryId)
FROM '/home/nkaramooz/Documents/alimbic/resources/snomed_files/Refset/Map/der2_iisssccRefset_ExtendedMapSnapshot_US1000124_20200901.txt'
WITH (FORMAT csv, HEADER true, DELIMITER '	', QUOTE E'\b');
