set schema 'ml2';
drop table if exists treatment_candidates;


create table treatment_candidates as (

	select distinct on (sentence_id, condition_acid, treatment_acid) sentence_id, condition_acid, treatment_acid, sentence_tuples, ver
	from (
 	select 
 		t1.sentence_id
 		,t1.condition_acid
 		,t3.treatment_acid
 		,t4.sentence_tuples
 		,0 as ver
 	from (select distinct sentence_id, acid as condition_acid from pubmed.sentence_annotations) t1
 	join ml2.cso t2 
 		on t1.condition_acid = t2.acid
 	join (select distinct sentence_id, acid as treatment_acid 
 			from pubmed.sentence_annotations 
 			where 
 				acid in (select child_acid from snomed2.transitive_closure_acid where parent_acid in ('165831', '250976', '92218', '18621', '233259'))) t3
 		on t1.sentence_id = t3.sentence_id
 	join pubmed.sentence_tuples t4
 		on t1.sentence_id = t4.sentence_id 
 		) t5

);

create index treatment_candidate_condition_acid_ind on treatment_candidates(condition_acid);
create index treatment_candidate_sentence_id_ind on treatment_candidates(sentence_id);
create index treatment_candidate_treatment_acid_ind on treatment_candidates(treatment_acid);