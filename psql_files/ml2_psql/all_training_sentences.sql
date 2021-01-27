set schema 'ml2';

drop table if exists all_training_sentences;

create table all_training_sentences as (
	select
		public.uuid_generate_v4()::text as id
		,t1.sentence_id
		,t2.section_ind
		,t1.x_train_gen
		,t1.x_train_mask
		,t1.condition_acid
		,t1.treatment_acid
		,t1.ver
		,t1.label
		,random() as rand
	from ml2.training_sentences_with_version t1
	join pubmed.sentence_concept_arr_1_8 t2
	on t1.sentence_id = t2.sentence_id
);

create index if not exists root_train_label_ind on all_training_sentences(label);
create index if not exists root_id_train_ind on all_training_sentences(id);