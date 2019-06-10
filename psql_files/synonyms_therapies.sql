set schema 'annotation';
drop table if exists synonyms_therapies;


create table synonyms_therapies as (
select 
	t1.id, t1.conceptid, t1.term || ' therapy' as new_term, t1.term as term
from annotation.active_cleaned_selected_concept_descriptions t1
join (select subtypeid from snomed.curr_transitive_closure_f where supertypeid = '373873005'
	or supertypeid = '105590001' or supertypeid = '71388002') t2
on t1.conceptid = t2.subtypeid
);


create index if not exists therapies_ind on synonyms_therapies(term);
create index if not exists conceptid_ind on synonyms_therapies(conceptid);