set schema 'annotation2';


drop table if exists add_adid_acronym;
create table add_adid_acronym (
	adid integer not null
    ,acid integer not null
	,term varchar(400) not null
	,word varchar(100) not null
	,word_ord integer not null
	,term_length integer not null
	,is_acronym boolean
);
create index concurrently add_adid_acronym_acid_ind on add_adid_acronym(acid);
create index concurrently add_adid_acronym_adid_ind on add_adid_acronym(adid);
create index concurrently add_term_acronym_term_ind on add_adid_acronym(term);
create index concurrently add_term_acronym_word_ind on add_adid_acronym(word);
create index concurrently add_term_acronym_word_ord_ind on add_adid_acronym(word_ord);


insert into add_adid_acronym
	select 
        concept_table.adid
        ,concept_table.acid
        ,concept_table.term
        ,concept_table.word
        ,concept_table.word_ord 
        ,len_tb.term_length
        ,case when acr.is_acronym is not NULL then acr.is_acronym else concept_table.is_acronym end as is_acronym 
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
                from annotation2.root_did
                ) tb, unnest(string_to_array(replace(replace(replace(replace(replace(term, ' - ', ' '), '.', ''), '-', ' '), ',', ''), '''', ''), ' '))
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
                    ,lower(unnest(string_to_array(replace(replace(replace(replace(replace(term, ' - ', ' '), '.', ''), '-', ' '), ',', ''), '''', ''), ' '))) as word
                from (
                    select 
                        adid
                        ,acid
                        ,term
                        ,active
                        ,row_number () over (partition by adid order by effectivetime desc) as row_num 
                    from annotation2.root_did
                    ) tb
                where row_num = 1 and active = '1'
                ) nm
            ) tmp
        group by tmp.adid
        ) len_tb
      on concept_table.adid = len_tb.adid
    left join (select t2.adid, t1.is_acronym from annotation2.acronym_override t1 
    		join annotation2.root_did t2 
    		on t1.did = t2.did) acr
      on concept_table.adid = acr.adid;