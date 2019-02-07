set schema 'annotation';

drop table if exists labelled_treatments_confirmation;
create table labelled_treatments_confirmation (
  condition_id varchar(40)
  ,treatment_id varchar(40)
  ,label integer
);

INSERT INTO annotation.labelled_treatments_confirmation (condition_id, treatment_id, label)
	VALUES
	('41291007', '37411004', 0),
	('89627008', '387025007', 0),
	('427399008', '69236009', 0),
	('450886002', '108809004', 0),
	('2429008', '77465005', 0)
	;


