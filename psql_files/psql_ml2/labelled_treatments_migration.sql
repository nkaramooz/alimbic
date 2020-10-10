set schema 'ml2';

insert into ml2.labelled_treatments_app
	select
		case when t4.condition_id='%' then '%' else t5.acid end as condition_acid
		,case when t4.treatment_id='%' then '%' else t6.acid end as treatment_id
		,label
	from (
		select 
			case when t1.condition_id = '%' then '%'
				when t2.syn_rank is not null and t2.syn_rank < t2.ref_rank then t2.syn_cid when t2.syn_rank is not null
					and t2.syn_rank=t2.ref_rank and t2.syn_cid < t2.ref_cid then t2.syn_cid else condition_id end as condition_id
			,case when t1.treatment_id = '%' then '%'
				when t3.syn_rank is not null and t3.syn_rank < t3.ref_rank then t3.syn_cid when t3.syn_rank is not null
					and t3.syn_rank=t3.ref_rank and t3.syn_cid < t3.ref_cid then t3.syn_cid else treatment_id end as treatment_id
			,label
		from annotation.labelled_treatments t1
		left join (select distinct on (ref_cid, ref_rank, syn_cid, syn_rank) ref_cid, ref_rank, syn_cid, syn_rank from annotation2.snomed_synonyms
			) t2
			on t1.condition_id = t2.ref_cid 
		left join (select distinct on (ref_cid, ref_rank, syn_cid, syn_rank) ref_cid, ref_rank, syn_cid, syn_rank from annotation2.snomed_synonyms
			) t3
			on t1.treatment_id = t3.ref_cid

		) t4
	left join annotation2.downstream_root_cid t5
		on t4.condition_id = t5.cid
	left join annotation2.downstream_root_cid t6
		on t4.treatment_id = t6.cid
