set schema 'annotation';
create table raw_treatment_recs(
  sentence text
  ,sentence_tuples text
  ,condition_id varchar(40)
  ,treatment_id varchar(40)
  ,score float
  ,pmid varchar(40)
  ,id varchar(40)
);