set schema 'annotation';
drop table if exists word_counts_50k;


create table word_counts_50k as (
    select t2.id, case when t2.rn <= 49996 then t2.word else 'UNK' end as word, t2.count, t2.rn, case when t2.rn <= 49996 then 1 else 0 end as grp 
	from (
		select t1.id, t1.word, t1.count, row_number() over(order by t1.count desc) + 1 as rn
		from annotation.word_counts t1
		order by t1.count desc
		) t2
);

create index if not exists word_50k_ind on word_counts_50k(word);

