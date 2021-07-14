set schema 'ml2';
drop table if exists treatment_candidates_1_9;

create table if not exists treatment_candidates_1_9 (
	sentence_id varchar(40) not null
	,condition_acid varchar(18) not null
	,treatment_acid varchar(18) not null
	,sentence_tuples json
	,pmid varchar(18)
	,year integer
	,ver integer
);

create index if not exists treatment_candidate_1_9_condition_acid_ind on treatment_candidates_1_9(condition_acid);
create index if not exists treatment_candidate_1_9_sentence_id_ind on treatment_candidates_1_9(sentence_id);
create index if not exists treatment_candidate_1_9_treatment_acid_ind on treatment_candidates_1_9(treatment_acid);

INSERT INTO treatment_candidates_1_9

	select distinct on (sentence_id, condition_acid, treatment_acid) 
		sentence_id, condition_acid, treatment_acid, sentence_tuples, pmid, year, ver
	from (
 	select 
 		t1.sentence_id
 		,t1.condition_acid
 		,t3.treatment_acid
 		,t4.sentence_tuples
 		,t4.pmid
 		,t4.journal_pub_year::int as year
 		,0 as ver
 	from (select distinct sentence_id, acid as condition_acid from pubmed.sentence_annotations_1_9 where ver=0) t1
 	join (select root_acid from annotation2.concept_types where rel_type='condition' or rel_type='symptom') t2 
 		on t1.condition_acid = t2.root_acid
 	join (select distinct sentence_id, acid as treatment_acid 
 			from pubmed.sentence_annotations_1_9 
 			where ver=0 and 
 				acid in (select root_acid from annotation2.concept_types where rel_type='treatment')) t3
 		on t1.sentence_id = t3.sentence_id
 	join pubmed.sentence_tuples_1_9 t4
 		on t1.sentence_id = t4.sentence_id 
 		) t5
;


-- update pubmed.sentence_tuples_1_9 set ver=1;
-- update pubmed.sentence_annotations_1_9 set ver=1;
-- update pubmed.sentence_concept_arr_1_9 set ver=1;


