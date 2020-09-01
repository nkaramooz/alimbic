set schema 'annotation2';

insert into snomed_synonyms
	select 
		t1.cid as ref_cid
	    ,t1.term as ref_term
	    ,case  
	    	when t3.supertypeid = '105590001' then 1
	        when t3.supertypeid = '373873005' then 2
	    	when t3.supertypeid = '71388002' then 3
	    	when t3.supertypeid = '404684003' then 4
	    	when t3.supertypeid = '123037004' then 5
	    	when t3.supertypeid = '363787002' then 6
	    	else 7 end as ref_rank
	    ,t2.cid as syn_cid
	    ,case 
	    	when t4.supertypeid = '105590001' then 1
	        when t4.supertypeid = '373873005' then 2 
	    	when t4.supertypeid = '71388002' then 3
	    	when t4.supertypeid = '404684003' then 4
	    	when t4.supertypeid = '123037004' then 5
	    	when t4.supertypeid = '363787002' then 6
	    	else 7 end as syn_rank
	from annotation2.cleaned_selected_descriptions_de_duped t1
	join annotation2.cleaned_selected_descriptions_de_duped t2
	  on t1.term = t2.term and t1.cid != t2.cid
	join snomed2.curr_transitive_closure_f t3
	on t1.cid = t3.subtypeid and t3.supertypeid in ('123037004', '404684003', '308916002',
		'272379006', '363787002', '410607006', '373873005', '78621006', '260787004',
		'71388002', '362981000', '105590001', '254291000', '123038009', '370115009', '48176007')
	join snomed2.curr_transitive_closure_f t4
	on t2.cid = t4.subtypeid and t4.supertypeid in ('123037004', '404684003', '308916002',
		'272379006', '363787002', '410607006', '373873005', '78621006', '260787004',
		'71388002', '362981000', '105590001', '254291000', '123038009', '370115009', '48176007')
	where t1.did not in (select did from annotation2.acronym_override)
;
