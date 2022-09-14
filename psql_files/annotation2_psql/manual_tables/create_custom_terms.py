import sys
sys.path.append('../../../utilities')
import pandas as pd
import pglib as pg

custom_term_synonym_dict = {
	'deficiency' : 'insufficiency'
	,'agent' : 'drug'
	,'antagonist' : 'blocker'
	,'antagonist' : 'antagonism'
	,'antagonist' : 'inhibitor'
	,'inhibitor' : 'inhibition'
	,'blocker' : 'blockade'
	,'blocker' : 'antagonist'
	,'regurgitation' : 'insufficiency'
	,'transplantation' : 'transplant'
	,'vagus' : 'vagal'
	,'and' : 'plus'
	,'hypokinetic' : 'hypokinesis'
	,'myeloid' : 'myelogenous'
	,'staphylococcus' : 'staph'
	,'streptococcus' : 'strep'
	,'examination' : 'exam'
	,'ultrasonography' : 'ultrasound'
	,'kidney' : 'renal'
	,'stroke' : 'CVA'
	,'alfa' : 'alpha'
	,'beta' : 'β'
	,'alpha' : 'α'
	,'leukaemia' : 'leukemia'
	,'anticoagulant' : 'AC'
	,'anticoagulation' : 'AC'
	,'pulmonary embolism' : 'PE'
	,'patent foramen ovale' : 'PFO'
	,'tumour necrosis factor' : 'TNF'
	,'tumor necrosis factor' : 'TNF'
	,'aortic valve' : 'AV'
	,'mitral valve' : 'MV'
	,'percutaneous coronary intervention' : 'PCI'
	,'angiotensin-converting enzyme' : 'ACE'
	,'superior mesenteric vein' : 'SMV'
	,'inferior mesenteric vein' : 'IMV'
	,'myocardial infarction' : 'MI'
	,'acute myocardial infarction' : 'AMI'
	,'heart' : 'cardiac'
	,'transplant' : 'transplantation'
	,'ulcer' : 'ulceration'
	,'venous thromboembolism' : 'VTE'
	,'deep venous thrombosis' : 'DVT'
	,'Epidermal growth factor receptor' : 'EGFR'
	,'clavulanic acid' : 'clavulanate'
	,'antimuscarinic drug' : 'muscarinic receptor antagonist'
	,'varicella-zoster' : 'VZV'
	,'echocardiography' : 'echocardiogram'
	,'receptor agonist' : 'agonist'
	,'Propionibacterium' : 'P.'
	,'glomerulonephritis' : 'glomerulopathy'
	,'refractory' : 'resistant'
	,'carcinoma' : 'cancer'
	,'HER2' : 'Human epidermal growth factor receptor 2'
	,'Needed' : 'Necessary'
	,'Multiple sclerosis' : 'MS'
	,'treatment' : 'remedy'
	,'Clostridium difficile' : 'C. diff'
	,'Schistosoma' : 'S.'
	,'prostate' : 'prostatic'
	,'ulcer' : 'ulceration'
	,'X-ray' : 'radiography'
	,'hypoplastic left heart' : 'HLH'
	,'Expressive language delay' : 'Delayed speech'
	,'Clostridium' : 'C.'
	,'Clostridium' : 'C'
}

def create_custom_phrases():
	conn,cursor = pg.return_postgres_cursor()

	for key,value in custom_term_synonym_dict.items():
		query = """
 			insert into annotation2.custom_terms
			select 
				public.uuid_generate_v4() as did
				,t2.acid
				,t2.term 
				,now() as effectivetime
			from 
				(select 
					acid
					,adid
					,regexp_replace(lower(term), concat(concat('\m', %s),'\M'), %s, 'g') as term
				from 
					(select 
						acid
						,adid
						,term 
					from annotation2.downstream_root_did 
					where acid in (select distinct(root_acid) from annotation2.concept_types where rel_type in ('condition', 'symptom', 'chemical', 'prevention', 'treatment', 'statistic', 'diagnostic', 'outcome', 'cause', 'anatomy') 
						and active != 0)
					and term ~* concat(concat('\m', %s), '\M')
					and term not ilike concat('%%', %s, '%%', %s)

				) t1 
			) t2 
			left join annotation2.downstream_root_did t3 
				on t2.acid=t3.acid and t2.term=t3.term where t3.term is null
			ON CONFLICT (acid, term) DO NOTHING
		"""
		cursor.execute(query, (key, value, key, key, key))
		cursor.connection.commit()
	cursor.close()
	conn.close()

def repair_of_phrases():
	conn,cursor = pg.return_postgres_cursor()
	query = """
		insert into annotation2.custom_terms
		select
			public.uuid_generate_v4() as did
			,t1.acid
			,t1.term
			,now() as effectivetime
		from (
			select
				acid 
				,concat(replace(lower(term), 'repair of ', ''), ' repair') as term
			from annotation2.downstream_root_did where term ilike 'repair of %'
		) t1
		left join (select term from annotation2.downstream_root_did) t2
			on t1.term = lower(t2.term)
		ON CONFLICT (acid, term) DO NOTHING
	"""
	cursor.execute(query, None)
	cursor.connection.commit()
	cursor.close()
	conn.close()


if __name__ == "__main__":
	create_custom_phrases()
	repair_of_phrases()
	# create_custom_terms()