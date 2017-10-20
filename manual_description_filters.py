import pandas as pd
import re
import psycopg2
from sqlalchemy import create_engine
from fuzzywuzzy import fuzz
import pglib as pg
import numpy as np
import time
import utils as u 


# engine = pg.return_sql_alchemy_engine()
	# results_df.to_sql('doc_annotation', engine, schema='annotation', if_exists='append')

DESC_ID_BLACKLIST = pd.DataFrame([ \
	['c86c4c65-a719-446a-9717-a62b1ac05cd4'] \
	], columns=['id'])
DESCRIPTION_WHITELIST = pd.DataFrame([ \
	['42343007', 'CHF'] \
	,['10784006', 'Antipsychotics']], columns=['conceptid', 'term'])


# NOTE: Still need to run psql file to update the
# augmented_concept_descriptions table based on 
# new filters
def update_filters_tables():
	engine = pg.return_sql_alchemy_engine()

	# NOTE - at some point write test such that blacklist and whitelist
	# don't collide
	DESC_ID_BLACKLIST.to_sql('description_id_blacklist', engine, schema='annotation', if_exists='replace')

	DESCRIPTION_WHITELIST.to_sql('description_whitelist', engine, schema='annotation', if_exists='replace')


if __name__ == "__main__":
	update_filters_tables()