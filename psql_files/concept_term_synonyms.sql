set schema 'annotation';
drop table if exists concept_terms_synonyms;


create table concept_terms_synonyms as (
select 
	t1.conceptid as reference_conceptid
    ,t1.term as reference_term
    ,case when t3.supertypeid = '373873005' then 1 
    	when t3.supertypeid = '105590001' then 2
    	when t3.supertypeid = '71388002' then 3
    	when t3.supertypeid = '404684003' then 4
    	when t3.supertypeid = '123037004' then 5
    	when t3.supertypeid = '363787002' then 6
    	else 7 end as reference_rank
    ,t2.conceptid as synonym_conceptid
    ,t2.term as synonym_term
    ,case when t4.supertypeid = '373873005' then 1 
    	when t4.supertypeid = '105590001' then 2
    	when t4.supertypeid = '71388002' then 3
    	when t4.supertypeid = '404684003' then 4
    	when t4.supertypeid = '123037004' then 5
    	when t4.supertypeid = '363787002' then 6
    	else 7 end as synonym_rank
from annotation.active_cleaned_selected_concept_descriptions t1
join annotation.active_cleaned_selected_concept_descriptions t2
  on t1.term = t2.term and t1.conceptid != t2.conceptid
join snomed.curr_transitive_closure_f t3
on t1.conceptid = t3.subtypeid and t3.supertypeid in ('123037004', '404684003', '308916002',
	'272379006', '363787002', '410607006', '373873005', '78621006', '260787004',
	'71388002', '362981000', '105590001', '254291000', '123038009', '370115009', '48176007')
join snomed.curr_transitive_closure_f t4
on t2.conceptid = t4.subtypeid and t4.supertypeid in ('123037004', '404684003', '308916002',
	'272379006', '363787002', '410607006', '373873005', '78621006', '260787004',
	'71388002', '362981000', '105590001', '254291000', '123038009', '370115009', '48176007')


union 

select 
	t2.conceptid as reference_conceptid
	,t2.term as reference_term
	,case when t4.supertypeid = '373873005' then 1 
		when t4.supertypeid = '105590001' then 2
    	when t4.supertypeid = '71388002' then 3
    	when t4.supertypeid = '404684003' then 4
    	when t4.supertypeid = '123037004' then 5
    	when t4.supertypeid = '363787002' then 6
    	else 7 end as reference_rank
	,t1.conceptid as synonym_conceptid
	,t1.term as synonym_term
	,case when t5.supertypeid = '373873005' then 1 
		when t5.supertypeid = '105590001' then 2
    	when t5.supertypeid = '71388002' then 3
    	when t5.supertypeid = '404684003' then 4
    	when t5.supertypeid = '123037004' then 5
    	when t5.supertypeid = '363787002' then 6
    	else 7 end as synonym_rank
from annotation.active_cleaned_selected_concept_descriptions t1
join (select subtypeid from snomed.curr_transitive_closure_f where supertypeid='71388002' 
	  or supertypeid='105590001' or supertypeid = '373873005') t3
on t1.conceptid = t3.subtypeid
join annotation.synonyms_therapies t2
on t1.term = t2.new_term and t1.conceptid != t2.conceptid
join snomed.curr_transitive_closure_f t4
on t2.conceptid = t4.subtypeid and t4.supertypeid in ('123037004', '404684003', '308916002',
	'272379006', '363787002', '410607006', '373873005', '78621006', '260787004',
	'71388002', '362981000', '105590001', '254291000', '123038009', '370115009', '48176007')
join snomed.curr_transitive_closure_f t5
on t1.conceptid = t5.subtypeid and t5.supertypeid in ('123037004', '404684003', '308916002',
	'272379006', '363787002', '410607006', '373873005', '78621006', '260787004',
	'71388002', '362981000', '105590001', '254291000', '123038009', '370115009', '48176007')

);

insert into concept_terms_synonyms
	select 
		t1.synonym_conceptid as reference_conceptid
		,t1.synonym_term as reference_term
		,t1.synonym_rank as reference_rank
		,t1.reference_conceptid as synonym_conceptid
		,t1.reference_term as synonym_term
		,t1.reference_rank as synonym_rank
	from annotation.concept_terms_synonyms t1;

create index if not exists syn_reference_conceptid on concept_terms_synonyms(reference_conceptid);
create index if not exists syn_reference_term on concept_terms_synonyms(reference_term);
create index if not exists syn_syn_conceptid on concept_terms_synonyms(synonym_conceptid);
create index if not exists syn_syn_term on concept_terms_synonyms(synonym_term);