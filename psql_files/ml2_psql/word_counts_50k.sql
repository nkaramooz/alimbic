set schema 'ml2';
drop table if exists word_counts_50k;

-- old 49996

create table word_counts_50k as (
    select case when t2.rn < 49999 then t2.concept else 'UNK' end as word, t2.rn, case when t2.rn < 49999 then 1 else 0 end as grp 
	from (
		select t1.concept, t1.cnt, row_number() over(order by t1.cnt desc) + 1 as rn
		from annotation2.concept_counts t1
		) t2
);

create index if not exists word_50k_ind on word_counts_50k(word);

