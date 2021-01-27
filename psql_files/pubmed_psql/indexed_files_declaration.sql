set schema 'pubmed';

drop table if exists indexed_files_1_8;
create table indexed_files_1_8 (
  filename varchar(40)
  ,file_num integer
);

create index indexed_files_filename_1_8_ind on indexed_files_1_8(filename);
create index indexed_files_file_num_1_8_ind on indexed_files_1_8(file_num);