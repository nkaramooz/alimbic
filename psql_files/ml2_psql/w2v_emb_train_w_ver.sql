create table ml2.w2v_emb_train_w_ver (
	sentence_id varchar(400) not null
	,x_train_spec jsonb
	,ver integer
);

insert into ml2.w2v_emb_train_w_ver
	select
		sentence_id
		,x_train_spec
		,0 as ver
	from ml2.w2v_emb_train
;

create index if not exists w2v_emb_train_w_ver_sentence_id_ind on ml2.w2v_emb_train_w_ver(sentence_id);
create index if not exists w2v_emb_train_w_ver_ver_ind on ml2.w2v_emb_train_w_ver(ver);