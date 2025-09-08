import glob
import pandas as pd
import numpy as np

# Find all files matching 'full_vec' in the current directory and subdirectories
full_vec_files = glob.glob('/home/data/yotam/full_vec_simulation/double*.csv', recursive=True)  + glob.glob('/home/data/yotam/full_vec_simulation/rotated*.csv',recursive = True)

# Read all files into a list of DataFrames
dfs = []
for file in full_vec_files:
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except Exception as e1:
        #print(f"Could not read {file}: {e1}")
        # Try loading from backup location
        import os
        backup_file = os.path.join("/home/data/yotam/full_vec_data_backup/", os.path.basename(file))
        try:
            df = pd.read_csv(backup_file)
            dfs.append(df)
            #print(f"Loaded backup for {file} from {backup_file}")
        except Exception as e2:
            pass
            #print(f"Could not read backup {backup_file}: {e2}")

# Concatenate all DataFrames
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    # Group by 'circuit' and 'p', summing all other columns
    grouped_df = combined_df.groupby(['circuit', 'p'], as_index=False).sum()
else:
    grouped_df = pd.DataFrame()  # Empty DataFrame if no files found

# grouped_df now contains the combined rows with the same circuit and p values
grouped_df['accepted_shots'] = grouped_df.shots - grouped_df.discards
grouped_df['P'] = (grouped_df.errors / grouped_df.accepted_shots)
grouped_df['rate'] = 1/(1 - grouped_df.discards / grouped_df.shots)
grouped_df['dP'] = ((1*(grouped_df.errors==0)+grouped_df.errors)**0.5/grouped_df.accepted_shots)
grouped_df['dp_log10'] = np.log10(1/grouped_df.accepted_shots)
grouped_df['eval_hours'] = grouped_df.eval_time / 3600 * 18
print(grouped_df[['p','P','shots','rate','discards','errors', 'dP', 'dp_log10', 'eval_hours']])
print(f"total shots={grouped_df.shots.sum()}, total cpu hours={grouped_df.eval_hours.sum()}")

grouped_df.to_csv("combined_vec_sim.csv", index = False)

import subprocess

try:
        result = subprocess.check_output("squeue -u yotam | grep R | wc -l", shell=True, text=True)
        num_running_slurm = int(result.strip())
        print(f"Number of running slurm instances: {num_running_slurm}")
except Exception as e:
        print(f"Could not determine number of running slurm instances: {e}")

