set schema 'annotation2';


drop table if exists add_adid_acronym;
create table add_adid_acronym (
	adid varchar(36) not null
    ,acid varchar(36) not null
	,term varchar(400) not null
	,word varchar(100) not null
	,word_ord integer not null
	,term_length integer not null
	,is_acronym boolean
);
create index add_adid_acronym_acid_ind on add_adid_acronym(acid);
create index add_adid_acronym_adid_ind on add_adid_acronym(adid);
create index add_term_acronym_term_ind on add_adid_acronym(term);
create index add_term_acronym_word_ind on add_adid_acronym(word);
create index add_term_acronym_word_ord_ind on add_adid_acronym(word_ord);


insert into add_adid_acronym
	select 
        concept_table.adid
        ,concept_table.acid
        ,concept_table.term
        ,concept_table.word
        ,concept_table.word_ord 
        ,len_tb.term_length
        ,case 
            when acr.is_acronym is not NULL then acr.is_acronym
            when term ~ '^\d+(\.\d+)?$' then 'f' 
            when term ~ '[0-9\."]+$' then 'f'
            else concept_table.is_acronym end as is_acronym 
    from (
        select 
            adid
            ,acid
            ,term
            ,case when upper(word) = word then word else lower(word) end as word
            ,word_ord
            ,case when upper(term) = term then 't' else 'f' end ::boolean as is_acronym
        from (
            select 
                adid
                ,acid
                ,term
                ,word
                ,row_number () over (partition by adid) as word_ord
            from (
                select 
                    adid
                    ,acid
                    ,term
                    ,active
                    ,row_number () over (partition by adid order by effectivetime desc) as row_num 
                from annotation2.downstream_root_did
                ) tb, unnest(string_to_array(regexp_replace(regexp_replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(term, ' - ', ' '), '.', ''), '- ', ' '), ' -', ' '), '-', ' '), ',', ''), '''', ''), '   ', ' '), '  ', ' '), '\s+$', ''), '^\s+', ''), ' '))
                with ordinality as f(word)
            where row_num = 1 and active = '1'
            ) nm
    ) concept_table
    join (
        select
          adid
          ,count(*) as term_length
        from (
            select 
                adid
                ,acid
                ,term
                ,word
            from (
                select 
                    adid
                    ,acid
                    ,term
                    ,lower(unnest(string_to_array(regexp_replace(regexp_replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(term, ' - ', ' '), '.', ''), '- ', ' '), ' -', ' '), '-', ' '), ',', ''), '''', ''), '   ', ' '), '  ', ' '), '\s+$', ''), '^\s+', ''), ' '))) as word
                from (
                    select 
                        adid
                        ,acid
                        ,term
                        ,active
                        ,row_number () over (partition by adid order by effectivetime desc) as row_num 
                    from annotation2.downstream_root_did
                    ) tb
                where row_num = 1 and active = '1'
                ) nm
            ) tmp
        group by tmp.adid
        ) len_tb
      on concept_table.adid = len_tb.adid
    left join (
            select 
                tb3.adid, tb1.is_acronym
            from annotation2.acronym_override tb1
            join (
                    select id, row_number() over (partition by adid order by effectivetime desc) as rn_num
                    from annotation2.acronym_override group by id
                ) tb2 
            on tb1.id = tb2.id
            join annotation2.downstream_root_did tb3
            on tb1.adid = tb3.adid
            where tb2.rn_num = 1
        ) acr
      on concept_table.adid = acr.adid;