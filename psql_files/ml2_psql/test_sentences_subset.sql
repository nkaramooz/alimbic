set schema 'ml2';
drop table if exists test_sentences_subset;


create table test_sentences_subset (
  id varchar(36)
  ,sentence_id varchar(36)
  ,section_ind integer
  ,x_train_gen jsonb
  ,label integer
  ,ver_gen integer
);

insert into ml2.test_sentences_subset
	select 
    	id
    	,sentence_id
    	,section_ind
    	,x_train_gen
    	,label
    	,ver_gen
    from ml2.test_sentences
    where label = 0
    order by random()
    limit 3000;

insert into ml2.test_sentences_subset 
	select 
    	id
    	,sentence_id
    	,section_ind
    	,x_train_gen
    	,label
    	,ver_gen
    from ml2.test_sentences
    where label = 1
    order by random()
    limit 2000;

create index if not exists test_sentences_subset_label_ind on test_sentences_subset(label);
create index if not exists test_sentences_subset_ver_gen_ind on test_sentences_subset(ver_gen);
create index if not exists test_sentences_subset_id_ind on test_sentences_subset(id);
