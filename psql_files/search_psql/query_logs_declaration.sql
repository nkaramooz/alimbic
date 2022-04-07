set schema 'search';

drop table if exists query_logs;
create table query_logs (
  ip_address varchar(40)
  ,query varchar(400)
  ,primary_cids jsonb
  ,expanded_query_acids jsonb
  ,unmatched_terms varchar(400)
  ,start_year varchar(4)
  ,end_year varchar(4)
  ,journals jsonb
  ,condition_json jsonb
  ,treatment_json jsonb
  ,diagnostic_json jsonb
  ,cause_json jsonb
  ,querytime timestamp
);


create index logs_ip_add_ind on query_logs(ip_address);
create index logs_query_ind on query_logs(query);
create index logs_querytime_ind on query_logs(querytime);