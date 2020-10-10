set schema 'annotation';

drop table if exists root_training_sentences;

create table root_training_sentences as (
	select
		public.uuid_generate_v4()::text as id,
		id as sentence_id,
		x_train_gen,
		label
	from annotation.training_sentences_with_version_v2
);

create index if not exists root_train_label_ind on root_training_sentences(label);
create index if not exists root_id_train_ind on root_training_sentences(id);