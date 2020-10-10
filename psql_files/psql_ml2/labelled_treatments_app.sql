set schema 'ml2';

drop table if exists labelled_treatments_app;
create table labelled_treatments_app (
  condition_acid varchar(40)
  ,treatment_acid varchar(40)
  ,label integer
);
