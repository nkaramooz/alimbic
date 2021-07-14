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
	,'blocker' : 'blockade'
	,'blocker' : 'antagonist'
	,'therapy' : 'agent'
	,'regurgitation' : 'insufficiency'
	,'transplantation' : 'transplant'
	,'vagus' : 'vagal'
	,'and' : 'plus'
	,'hypokinetic' : 'hypokinesis'
	,'with' : 'and'
	,'and' : 'with'
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
	,'angiotensin converting enzyme' : 'ACE'
	,'superior mesenteric vein' : 'SMV'
	,'inferior mesenteric vein' : 'IMV'
	,'myocardial infarction' : 'MI'
	,'acute myocardial infarction' : 'AMI'
	,'heart' : 'cardiac'
	,'transplant' : 'transplantation'
	,'blocker' : 'blockade'
	,'ulcer' : 'ulceration'
	,'venous thromboembolism' : 'VTE'
	,'deep venous thrombosis' : 'DVT'
	,'Epidermal growth factor receptor' : 'EGFR'
	,'clavulanic acid' : 'clavulanate'

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
					,replace(lower(term), %s, %s) as term
				from 
					(select 
						acid
						,adid
						,term 
					from annotation2.downstream_root_did 
					where term ilike concat('%%',%s, '%%')
				) t1 
				where term not ilike concat('%%', %s, '%%')) t2 
			left join annotation2.downstream_root_did t3 
				on t2.acid=t3.acid and t2.term=t3.term where t3.term is null
			ON CONFLICT (acid, term) DO NOTHING
		"""
		cursor.execute(query, (key,value,key,value))
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

# def create_custom_terms():
# 	conn,cursor = pg.return_postgres_cursor()

# 	for key,value in custom_word_synonym_dict.items():
# 		query = """
# 		insert into annotation2.custom_terms
# 			select 
# 				t1.did
# 				,t2.acid
# 				,replace(lower(t2.term), %s, %s) as term
# 				,now() as effectivetime
# 			from (
# 				select 
# 					public.uuid_generate_v4() as did
# 					,adid as adid
# 				from annotation2.lemmas t1
# 				where t1.word = %s
# 			) t1
# 			join annotation2.lemmas t2
# 			on t1.adid = t2.adid
# 		ON CONFLICT (acid, term) DO NOTHING
# 		"""
# 		cursor.execute(query, (key, value, key))
# 		cursor.connection.commit()
# 	cursor.close()
# 	conn.close()

# below will only work on descendants of treatments
# Need relationship by acid built first

if __name__ == "__main__":
	create_custom_phrases()
	repair_of_phrases()
	# create_custom_terms()