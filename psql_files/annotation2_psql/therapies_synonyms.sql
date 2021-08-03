set schema 'annotation';

insert into therapies_synonyms
	select 
		t1.cid, t1.did, t1.term || ' therapy' as term, now() as effectivetime
	from annotation.cleaned_selected_descriptions_de_duped t1
	join (select subtypeid from snomed2.curr_transitive_closure_f where supertypeid = '373873005'
		or supertypeid = '105590001' or supertypeid = '71388002') t2
		on t1.cid = t2.subtypeid
	where t1.cid not in (select cid from annotation.snomed_cid_ignore)
	ON CONFLICT DO NOTHING
;


