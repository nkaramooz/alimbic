set schema 'annotation';
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

create table custom_lemmas as (

	select 
		new_id as description_id
		,conceptid
	    ,term
	    ,case when word='blocker' then 'blockade' else word end
	    ,word_ord
	    ,term_length
	from annotation.lemmas_3 l
	join (
		select 
			distinct on (description_id)
            description_id as old_id
			,public.uuid_generate_v4() as new_id
		from annotation.lemmas_3
		where description_id in (select description_id from annotation.lemmas_3 where word='blocker')
		) tb 
	on l.description_id = tb.old_id
	where description_id in (select description_id from annotation.lemmas_3 where word='blocker')
);

create index custom_lemma_did on custom_lemmas(description_id);
create index custom_lemma_cid on custom_lemmas(conceptid);
create index custom_lemma_term on custom_lemmas(term);
create index custom_lemma_word on custom_lemmas(word);
