import pandas as pd
import hashlib
from json import loads
from glob import glob
from tqdm import tqdm

def get_c(json_metadata):
    return loads(json_metadata)['c']

for filename in tqdm(list(glob("erasure_stats/erasure_stats*.csv"))):
    if "fixed_ids" in filename:
        continue
    df = pd.read_csv(filename)
    new_filename = filename.split(".csv")[0] + "_fixed_ids.csv"
    fltr = df.json_metadata.apply(get_c) == 'erasure'
    df.loc[fltr, 'strong_id'] = df[fltr].json_metadata.apply(lambda x:hashlib.sha256(x.encode()).digest().hex())
    df.to_csv(new_filename)
