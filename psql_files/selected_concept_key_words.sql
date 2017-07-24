set schema 'annotation';

drop table if exists selected_concept_key_words;
create table selected_concept_key_words as (


    select 
        concept_table.*
        ,len_tb.term_length
    from (
        select 
    		description_id
            ,conceptid
        	,term
            ,lower(word) as word
            ,word_ord
    	from (
        	select 
            	id as description_id
                ,conceptid
                ,term
                ,word
                ,word_ord
        	from annotation.selected_concept_descriptions, unnest(string_to_array(term, ' '))
                with ordinality as f(word, word_ord)
        	) nm
    	where lower(word) not in (select lower(words) from annotation.filter_words)
    ) concept_table
    join (
        select
          description_id
          ,count(*) as term_length
        from (
            select 
                description_id
                ,conceptid
                ,term
                ,word
            from (
                select 
                    id as description_id
                    ,conceptid
                    ,term
                    ,lower(unnest(string_to_array(term, ' '))) as word
                from annotation.selected_concept_descriptions
                ) nm
            where lower(word) not in (select lower(words) from annotation.filter_words)

            ) tmp
        group by tmp.description_id
        ) len_tb
      on concept_table.description_id = len_tb.description_id


);