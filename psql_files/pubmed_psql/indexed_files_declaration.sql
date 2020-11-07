set schema 'pubmed';

drop table if exists indexed_files;
create table indexed_files (
  filename varchar(40)
  ,file_num integer
);

create index indexed_files_filename_ind on indexed_files(filename);
create index indexed_files_file_num_ind on indexed_files(file_num);