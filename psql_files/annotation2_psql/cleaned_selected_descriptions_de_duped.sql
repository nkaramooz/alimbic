set schema 'annotation';

insert into cleaned_selected_descriptions_de_duped
	select
		did
		,cid
		,term
		,'1900-09-08 19:58:14.190442' as effectivetime
	from (
		select 
			tb1.did
			,tb1.cid
			,tb1.term
			,case when tb1.term != tb2.term and tb3.count > 1 then 0 else 1 end as keep
		from annotation.cleaned_selected_descriptions_prelim tb1
		join snomed2.active_selected_descriptions tb2
		on tb1.did = tb2.did
		join (select cid, term, count(*) from annotation.cleaned_selected_descriptions_prelim group by cid,term) tb3
		on tb1.cid = tb3.cid and tb1.term = tb3.term
	) tb4
	where tb4.keep = 1
	ON CONFLICT DO NOTHING;

-- default time is set to old so that manual overrides aren't replaced everytime SNOMED is updated.