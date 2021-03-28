set schema 'annotation';
drop table if exists word_counts_20k;


create table word_counts_20k as (
    select case when t2.rn <= 19996 then t2.concept else 'UNK' end as word, t2.rn, case when t2.rn <= 19996 then 1 else 0 end as grp 
	from (
		select t1.concept, t1.cnt, row_number() over(order by t1.cnt desc) + 1 as rn
		from annotation2.concept_counts t1
		) t2
);

create index if not exists word_20k_ind on word_counts_20k(word);

