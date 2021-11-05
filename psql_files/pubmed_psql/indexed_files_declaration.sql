set schema 'pubmed';

drop table if exists indexed_files_2;
create table indexed_files_2 (
  filename varchar(40)
  ,file_num integer
);

create index indexed_files_filename_2__ind on indexed_files_2(filename);
create index indexed_files_file_num_2__ind on indexed_files_2(file_num);