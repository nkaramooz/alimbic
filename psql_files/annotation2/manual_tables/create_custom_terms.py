import sys
sys.path.append('../../../utilities')
import pandas as pd
import pglib as pg

custom_synonym_dict = {
	'deficiency' : 'insufficiency'
	,'agent' : 'drug'
	,'antagonist' : 'blocker'
	,'antagonist' : 'antagonism'
	,'blocker' : 'blockade'
	,'blocker' : 'antagonist'
	,'regurgitation' : 'insufficiency'
	,'transplantation' : 'transplant'
	,'vagus' : 'vagal'
}

def create_custom_terms():
	conn,cursor = pg.return_postgres_cursor()

	for key,value in custom_synonym_dict.items():
		query = """
		insert into annotation2.custom_terms
			select 
				t1.did
				,t2.acid
				,replace(lower(t2.term), %s, %s) as term
				,now() as effectivetime
			from (
				select 
					public.uuid_generate_v4() as did
					,adid as adid
				from annotation2.add_adid_acronym t1
				where t1.word = 'deficiency'
			) t1
			join annotation2.add_adid_acronym t2
			on t1.adid = t2.adid
		ON CONFLICT (acid, term) DO NOTHING
		"""
		cursor.execute(query, (key, value))
		cursor.connection.commit()

# below will only work on descendants of treatments
# Need relationship by acid built first

if __name__ == "__main__":
	create_custom_terms()