set schema 'annotation';

drop table if exists labelled_treatments_confirmation;
create table labelled_treatments_confirmation (
  condition_id varchar(40)
  ,treatment_id varchar(40)
  ,label integer
);

INSERT INTO annotation.labelled_treatments_confirmation (condition_id, treatment_id, label)
	VALUES
	('19030005', '278910002', 0),
	('19030005', '386895008', 1),
	('19030005', '387105006', 1),
	('19030005', '387105006', 1)
	;


