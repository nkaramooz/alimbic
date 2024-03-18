set schema 'spacy';

drop table if exists spacy_primary_entities;
create table spacy_primary_entities (
    a_cid varchar(36) not null
	,term varchar(400) not null
	,description varchar(800) not null
	,cnt integer
	,unique(a_cid)
);

insert into spacy_primary_entities 
	select
		a_cid
		,term
		,description
		,case when t2.cnt is null then 0 else t2.cnt end
	from spacy.wiki_spacy_entities t1
	left join annotation2.concept_counts t2
	on t1.a_cid = t2.concept 
	where a_cid in (select root_acid from annotation2.concept_types
		where rel_type in ('condition', 'chemical', 'treatment', 'outcome', 'statistic', 'symptom', 'diagnostic', 'cause', 'anatomy', 'study_design')
		and (active=1 or active=3))
	ON CONFLICT(a_cid) DO NOTHING
;

insert into spacy_primary_entities 
	select
		a_cid
		,term
		,term as description
		,case when t2.cnt is null then 0 else t2.cnt end
		from (
			select 
				acid as a_cid
				,term
				,row_number () over (partition by acid order by length(term) desc) as rn_num
			from annotation2.downstream_root_did
			where acid in (
				select root_acid from annotation2.concept_types
				where rel_type in ('condition','chemical', 'treatment', 'outcome', 'statistic', 'symptom', 'diagnostic', 'cause','anatomy', 'study_design')
				and (active=1 or active=3)
			)
		) t1
	left join annotation2.concept_counts t2 
	on t1.a_cid = t2.concept 
	where rn_num = 1
ON CONFLICT(a_cid) DO NOTHING;

create index primary_entities_a_cid_ind on spacy_primary_entities(a_cid);