from earthdata import Auth, DataGranules, DataCollections, Store
from pprint import pprint
import requests


auth = Auth().login(strategy="netrc")

Query = DataGranules().short_name("TELLUS_GRAC_L3_CSR_RL06_LND_v04")

granules = Query.get()

files = Store(auth).get(granules, local_path='./data', )
