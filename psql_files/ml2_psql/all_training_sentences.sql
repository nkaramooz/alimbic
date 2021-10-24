set schema 'ml2';

create index if not exists training_sentences_with_version_1_9_condition_acid_ind on ml2.training_sentences_staging(condition_acid);
create index if not exists training_sentences_with_version_1_9_treatment_acid_ind on ml2.training_sentences_staging(treatment_acid);
create index if not exists training_sentences_with_version_1_9_sentence_id_ind on ml2.training_sentences_staging(sentence_id);


drop table if exists all_training_sentences;

create table all_training_sentences as (
	select
		distinct on (t1.condition_acid, t1.treatment_acid, t1.sentence_id)
		public.uuid_generate_v4()::text as id
		,t1.sentence_id
		,t1.sentence_tuples
		,t2.section_ind
		,t1.x_train_gen
		,t1.x_train_spec
		,t1.x_train_gen_mask
		,t1.x_train_spec_mask
		,t1.condition_acid
		,t1.treatment_acid
		,t1.og_condition_acid
		,t1.og_treatment_acid
		,t1.ver
		,t1.label
		,random() as rand
	from ml2.training_sentences_staging t1
	join pubmed.sentence_concept_arr_1_9 t2
	on t1.sentence_id = t2.sentence_id
);

-- insert into ml2.all_training_sentences
-- 	select
-- 		distinct on (t1.condition_acid, t1.treatment_acid, t1.sentence_id)
-- 		public.uuid_generate_v4()::text as id
-- 		,t1.sentence_id
-- 		,t1.sentence_tuples
-- 		,t2.section_ind
-- 		,t1.x_train_gen
-- 		,t1.x_train_spec
-- 		,t1.x_train_gen_mask
-- 		,t1.x_train_spec_mask
-- 		,t1.condition_acid
-- 		,t1.treatment_acid
-- 		,t1.og_condition_acid
-- 		,t1.og_treatment_acid
-- 		,t1.ver
-- 		,t1.label
-- 		,random() as rand
-- 	from ml2.training_sentences_staging t1
-- 	join pubmed.sentence_concept_arr_1_9 t2
-- 	on t1.sentence_id = t2.sentence_id
-- 	ON CONFLICT DO NOTHING
-- ;
	

create index if not exists root_train_label_ind on all_training_sentences(label);
create index if not exists root_id_train_ind on all_training_sentences(id);