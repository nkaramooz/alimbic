set schema 'snomed';

drop table if exists metadata_concept_key_words;

create table metadata_concept_key_words as (
	
    select 
        concept_table.conceptid
        ,concept_table.word
        ,description.term as term
    from (
        select 
            distinct on (conceptid, word)
    		conceptid
        	,word
    	from (
        	select 
            	conceptid
            	,lower(unnest(string_to_array(term, ' '))) as word
        	from snomed.metadata_concept_names
        	) nm
    	where lower(word) not in (select lower(words) from snomed.filter_words)
    ) concept_table
    join snomed.metadata_concept_names description 
        on description.conceptid = concept_table.conceptid
);