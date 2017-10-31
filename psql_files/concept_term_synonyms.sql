set schema 'annotation';
drop table if exists concept_terms_synonyms;


create table concept_terms_synonyms as (
select 
	t1.conceptid as reference_conceptid
    ,t1.term as reference_term
    ,t2.conceptid as synonym_conceptid
    ,t2.term as synonym_term
from annotation.active_cleaned_selected_concept_descriptions t1
join annotation.active_cleaned_selected_concept_descriptions t2
  on t1.term = t2.term and t1.conceptid != t2.conceptid

);