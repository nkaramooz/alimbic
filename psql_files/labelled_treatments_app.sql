set schema 'annotation';

drop table if exists labelled_treatments_app;
create table labelled_treatments_app (
  condition_id varchar(40)
  ,treatment_id varchar(40)
  ,label integer
);


