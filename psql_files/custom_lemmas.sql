set schema 'annotation';
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- # 'agent' -> 'drug' for pharmaceutic product ('373873005')
-- # 'antagonist' -> 'blocker' for all -- DONE
-- # 'antagonist' -> 'antagonism' -- DONE
-- # 'blocker' -> 'blockade'
-- # 'regurgitation' -> 'insufficiency'
-- # 'deficiency' -> 'insufficiency'

insert into annotation.lemmas_3
	select description_id, conceptid, term, term_lower, word, word_ord, term_length, is_acronym
	from annotation.custom_lemmas
	where description_id::text NOT IN (select description_id::text from annotation.lemmas_3)

-- insert into annotation.custom_lemmas

-- select t1.description_id, t2.conceptid, replace(t2.term, 'deficiency', 'insufficiency') as term
-- 		,replace(t2.term_lower, 'deficiency', 'insufficiency') as term_lower
-- 		,case when t2.word = 'deficiency' then 'insufficiency' else t2.word end
-- 		,t2.word_ord
-- 		,t2.term_length
-- 		,t2.is_acronym
-- 	from (
-- 		select public.uuid_generate_v4() as description_id, description_id as description_id2
-- 		from annotation.lemmas_3 t1
-- 		where t1.word = 'deficiency' and t1.conceptid in (select subtypeid from snomed.curr_transitive_closure_f )
-- 	) t1
-- 	join annotation.lemmas_3 t2
-- 	on t1.description_id2 = t2.description_id

-- select t1.description_id, t2.conceptid, replace(t2.term, 'regurgitation', 'insufficiency') as term
-- 		,replace(t2.term_lower, 'regurgitation', 'insufficiency') as term_lower
-- 		,case when t2.word = 'regurgitation' then 'insufficiency' else t2.word end
-- 		,t2.word_ord
-- 		,t2.term_length
-- 		,t2.is_acronym
-- 	from (
-- 		select public.uuid_generate_v4() as description_id, description_id as description_id2
-- 		from annotation.lemmas_3 t1
-- 		where t1.word = 'regurgitation' and t1.conceptid in (select subtypeid from snomed.curr_transitive_closure_f )
-- 	) t1
-- 	join annotation.lemmas_3 t2
-- 	on t1.description_id2 = t2.description_id

	-- select t1.description_id, t2.conceptid, replace(t2.term, 'agent', 'drug') as term
	-- 	,replace(t2.term_lower, 'agent', 'drug') as term_lower
	-- 	,case when t2.word = 'agent' then 'drug' else t2.word end
	-- 	,t2.word_ord
	-- 	,t2.term_length
	-- 	,t2.is_acronym
	-- from (
	-- 	select public.uuid_generate_v4() as description_id, description_id as description_id2
	-- 	from annotation.lemmas_3 t1
	-- 	where t1.word = 'agent' and t1.conceptid in (select subtypeid from snomed.curr_transitive_closure_f where supertypeid='373873005')
	-- ) t1
	-- join annotation.lemmas_3 t2
	-- on t1.description_id2 = t2.description_id

	-- select 
	-- 	new_id as description_id
	-- 	,conceptid
	--     ,term
	--     ,case when word='blocker' then 'blockade' else word end
	--     ,word_ord
	--     ,term_length
	-- from annotation.lemmas_3 l
	-- join (
	-- 	select 
	-- 		distinct on (description_id)
 --            description_id as old_id
	-- 		,public.uuid_generate_v4() as new_id
	-- 	from annotation.lemmas_3
	-- 	where description_id in (select description_id from annotation.lemmas_3 where word='blocker')
	-- 	) tb 
	-- on l.description_id = tb.old_id
	-- where description_id in (select description_id from annotation.lemmas_3 where word='blocker')


-- create index custom_lemma_did on custom_lemmas(description_id);
-- create index custom_lemma_cid on custom_lemmas(conceptid);
-- create index custom_lemma_term on custom_lemmas(term);
-- create index custom_lemma_word on custom_lemmas(word);
