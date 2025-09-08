import glob
import stim
import sys
sys.path.append('./cultiv_code/src')
from cultiv import ErrorEnumerationReport
import glob
import stim
from tqdm import tqdm


d3_circuits = glob.glob("./circuits/HCultivationSurfaceCode/c=init-H*.stim")  
def write_cache_file(cache_file):
    with open("./error_cache.txt", "w") as f:
        for key, value in cache_file.items():
            f.write(f"ENTRY {key}\n")
            for error in value:
                f.write(f"{','.join(map(str, error))}\n")

if __name__ == "__main__":  
    cache_file = ErrorEnumerationReport.read_cache_file("./error_cache.txt")
    for filename in tqdm(d3_circuits):
        circ = stim.Circuit.from_file(filename)
        report = ErrorEnumerationReport.from_circuit(circ, max_weight=4, cache=cache_file)
        write_cache_file(cache_file)

    d5_circuits = glob.glob("./circuits/HCultivationSurfaceCode/c=expansion*.stim")  
    from joblib import Parallel, delayed

    def process_circuit(filename, cache_file):
        circ = stim.Circuit.from_file(filename)
        report = ErrorEnumerationReport.from_circuit(circ, max_weight=5, cache=cache_file)
        return cache_file

        # Run in parallel
    results = Parallel(n_jobs=-1)(
            delayed(process_circuit)(filename, cache_file) for filename in d5_circuits
        )

        # Merge all cache files
    merged_cache = {}
    for res in results:
        merged_cache |= res

    old_cache_file = ErrorEnumerationReport.read_cache_file("./error_cache.txt")
    write_cache_file(old_cache_file | merged_cache)
