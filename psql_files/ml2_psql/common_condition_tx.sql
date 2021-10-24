set schema 'ml2';
drop table if exists common_condition_tx;

create table if not exists common_condition_tx (
	condition_acid varchar(18) not null
	,treatment_acid varchar(18) not null
	,cnt integer
);

create index if not exists common_condition_tx_condition_acid_ind on common_condition_tx(condition_acid);
create index if not exists common_condition_tx_treatment_acid_ind on common_condition_tx(treatment_acid);

INSERT INTO common_condition_tx

	select
		condition_acid
		,treatment_acid
		,count(*) as cnt
	from (
 	select 
 		t1.condition_acid
 		,t3.treatment_acid
 	from (select distinct sentence_id, acid as condition_acid from pubmed.sentence_annotations_1_9) t1
 	join (
 			select root_acid 
 			from annotation2.concept_types where (rel_type='condition' or rel_type='symptom') and active=1
 		) t2 
 	on t1.condition_acid = t2.root_acid
 	join (
 			select distinct sentence_id, acid as treatment_acid 
 			from pubmed.sentence_annotations_1_9 
 			where acid in (select root_acid from annotation2.concept_types where rel_type='treatment' and active=1)
 		) t3
 	on t1.sentence_id = t3.sentence_id

	) t5
	group by condition_acid, treatment_acid
;


-- update pubmed.sentence_tuples_1_9 set ver=1;
-- update pubmed.sentence_annotations_1_9 set ver=1;
-- update pubmed.sentence_concept_arr_1_9 set ver=1;