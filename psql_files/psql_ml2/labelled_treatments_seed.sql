set schema 'ml2';

drop table if exists labelled_treatments_seed;
create table labelled_treatments_seed (
  condition_acid varchar(40)
  ,treatment_acid varchar(40)
  ,label integer
);

-- value of 2 = too broad or partial treatment term
-- will not be displayed in app

-- INSERT INTO annotation.labelled_treatments_seed(condition_acid, treatment_acid, label)
-- 	;
