set schema 'annotation';
drop table if exists concept_terms_synonyms;


create table concept_terms_synonyms as (
    select distinct reference_conceptid, reference_term, reference_rank, synonym_conceptid, synonym_term, synonym_rank
    from annotation.concept_terms_synonyms_prelim
);



create index if not exists syn_reference_conceptid on concept_terms_synonyms(reference_conceptid);
create index if not exists syn_reference_term on concept_terms_synonyms(reference_term);
create index if not exists syn_syn_conceptid on concept_terms_synonyms(synonym_conceptid);
create index if not exists syn_syn_term on concept_terms_synonyms(synonym_term);