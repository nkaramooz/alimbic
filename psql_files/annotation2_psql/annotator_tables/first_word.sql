set schema 'annotation2';
drop table if exists first_word;

create table first_word as (
	select 
		word, 
		array_agg(adid) as adid_agg
	from annotation2.lemmas
	where word_ord = 1
	group by word
		
);

create index word on first_word(word);
