set schema 'annotation';

drop table if exists augmented_active_key_words_v3;
create table augmented_active_key_words_v3 as (


   select 
        concept_table.*
        ,len_tb.term_length
    from (
        select 
            description_id
            ,conceptid
            ,term
            ,case when upper(word) = word then word else lower(word) end as word
            ,word_ord
        from (
            select 
                description_id
                ,conceptid
                ,term
                ,word
                ,row_number () over (partition by description_id) as word_ord
            from (
                select 
                    description_id
                    ,conceptid
                    ,term
                    ,active
                    ,row_number () over (partition by description_id order by effectivetime desc) as row_num 
                from annotation.augmented_selected_concept_descriptions

                ) tb, unnest(string_to_array(replace(replace(replace(replace(term, ' - ', ' '), '-', ' '), ',', ''), '''', ''), ' '))
                with ordinality as f(word)
            where row_num = 1 and active = '1'
            ) nm
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
                    description_id
                    ,conceptid
                    ,term
                    ,lower(unnest(string_to_array(replace(replace(replace(replace(term, ' - ', ' '), '-', ' '), ',', ''), '''', ''), ' '))) as word
                from (
                    select 
                        description_id
                        ,conceptid
                        ,term
                        ,active
                        ,row_number () over (partition by description_id order by effectivetime desc) as row_num 
                    from annotation.augmented_selected_concept_descriptions

                    ) tb
                where row_num = 1 and active = '1'
                ) nm
            ) tmp
        group by tmp.description_id
        ) len_tb
      on concept_table.description_id = len_tb.description_id


);

create index kw3_conceptid on augmented_active_key_words_v3(conceptid);
create index kw3_description_id on augmented_active_key_words_v3(description_id);
create index kw3_term on augmented_active_key_words_v3(term);
create index kw3_word on augmented_active_key_words_v3(word);
create index kw3_word_ord on augmented_active_key_words_v3(word_ord);