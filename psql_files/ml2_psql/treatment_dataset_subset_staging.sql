set schema 'ml2';
drop table if exists treatment_dataset_subset_staging;

create index treatment_dataset_staging_entry_id on treatment_dataset_staging(entry_id);

create table if not exists treatment_dataset_subset_staging (
	entry_id varchar(40) not null
	,condition_acid varchar(18) not null
	,treatment_acid varchar(18) not null
	,sentence_tuples json
	,x_train_gen json
	,ver integer
);

create index if not exists treatment_dataset_subset_staging_condition_acid_ind on treatment_dataset_subset_staging(condition_acid);
create index if not exists treatment_dataset_subset_staging_entry_id_ind on treatment_dataset_subset_staging(entry_id);
create index if not exists treatment_dataset_subset_staging_treatment_acid_ind on treatment_dataset_subset_staging(treatment_acid);


INSERT INTO treatment_dataset_subset_staging
	select
		entry_id
		,condition_acid
		,treatment_acid 
		,sentence_tuples
		,x_train_gen
		,ver
	from (
		select 
			entry_id
			,condition_acid
			,treatment_acid 
			,sentence_tuples 
			,x_train_gen
			,ver
			,row_number() over (partition by condition_acid, treatment_acid) as rn_num
		from ml2.treatment_dataset_staging
		) t1
	where rn_num <= 10
;