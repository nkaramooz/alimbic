set schema 'snomed2';

drop table if exists curr_transitive_closure_f;
create table curr_transitive_closure_f(
  subtypeId varchar(18) not null,
  supertypeId varchar(18) not null
);

COPY curr_transitive_closure_f(subtypeId, supertypeId)
FROM '/home/nkaramooz/Documents/alimbic/resources/snomed_files/transitive_closure.txt' 
WITH (FORMAT csv, HEADER false, DELIMITER '	', QUOTE E'\b');