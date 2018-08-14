set schema 'annotation';
create table annotation_errors(
  root_cid text
  ,associated_cid text
  ,effectivetime timestamp
);