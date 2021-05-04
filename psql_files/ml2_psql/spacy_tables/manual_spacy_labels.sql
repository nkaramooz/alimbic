set schema 'ml2';

drop table if exists manual_spacy_labels;

create table manual_spacy_labels (
	entry_id varchar(40)
	,sentence_id varchar(40)
  ,sentence_tuples jsonb
  ,label integer
  ,spacy_label jsonb
  ,ver integer
);

insert into ml2.manual_spacy_labels
	select
		public.uuid_generate_v4()::text as entry_id
		,sentence_id
		,sentence_tuples
		,label
		,NULL as spacy_label
		,0 as ver
	from ml2.spacy_training_data
;
-- create index if not exists root_train_label_ind on all_training_sentences(label);
-- create index if not exists root_id_train_ind on all_training_sentences(id);