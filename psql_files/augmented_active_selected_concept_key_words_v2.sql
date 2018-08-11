set schema 'annotation';

drop table if exists augmented_active_selected_concept_key_words_v2;
create table augmented_active_selected_concept_key_words_v2 as (


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
            where row_num = 1 and active = '1' and lower(word) not in (select lower(words) from annotation.filter_words)
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
            where lower(word) not in (select lower(words) from annotation.filter_words)

            ) tmp
        group by tmp.description_id
        ) len_tb
      on concept_table.description_id = len_tb.description_id


);

create index ascd_kw_conceptid on augmented_active_selected_concept_key_words_v2(conceptid);
create index ascd_kw_description_id on augmented_active_selected_concept_key_words_v2(description_id);
create index ascd_kw_term on augmented_active_selected_concept_key_words_v2(term);
create index ascd_kw_word on augmented_active_selected_concept_key_words_v2(word);
create index ascd_kw_word_ord on augmented_active_selected_concept_key_words_v2(word_ord);