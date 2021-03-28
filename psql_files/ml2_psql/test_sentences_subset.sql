set schema 'ml2';
drop table if exists test_sentences_subset;


create table test_sentences_subset (
  id varchar(36)
  ,sentence_id varchar(36)
  ,sentence_tuples jsonb
  ,condition_acid varchar(36)
  ,treatment_acid varchar(36)
  ,section_ind integer
  ,x_train_gen jsonb
  ,x_train_spec jsonb
  ,x_train_mask jsonb
  ,label integer
  ,ver integer
);

insert into ml2.test_sentences_subset
	select 
    	id
    	,sentence_id
      ,sentence_tuples
      ,condition_acid
      ,treatment_acid
    	,section_ind
    	,x_train_gen
      ,x_train_spec
      ,x_train_mask
    	,label
    	,ver
    from ml2.test_sentences
    where label = 0
    order by random()
    limit 3000;

insert into ml2.test_sentences_subset 
	select 
    	id
    	,sentence_id
      ,sentence_tuples
      ,condition_acid
      ,treatment_acid
    	,section_ind
    	,x_train_gen
      ,x_train_spec
      ,x_train_mask
    	,label
    	,ver
    from ml2.test_sentences
    where label = 1
    order by random()
    limit 3000;

create index if not exists test_sentences_subset_label_ind on test_sentences_subset(label);
create index if not exists test_sentences_subset_ver_gen_ind on test_sentences_subset(ver);
create index if not exists test_sentences_subset_id_ind on test_sentences_subset(id);
