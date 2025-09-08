from __future__ import annotations

import sys
sys.path.append("../StateVecSimulator/latte")
sys.path.append("../StateVecSimulator")

import numpy as np

import time
import simulation
import importlib
import _vec_intercept_sampler
import _noise
importlib.reload(simulation)
importlib.reload(_vec_intercept_sampler)
importlib.reload(_noise)
from _vec_intercept_sampler import VecInterceptSampler
from _noise import NoiseModel
from simulation import *
from IPython import display
import matplotlib.pyplot as plt
import stim
import sinter
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


simulation = Simulation(d = 3)

ghz_circuit = simulation.generate_projection_circuit(generate_unitary = False)
double_ghz_circuit = simulation.generate_double_ghz(generate_unitary = False)
syndrome_circuit = simulation.generate_syndrome_circuit(generate_unitary = False)
injection_circuit = simulation.generate_injection_circuit(generate_unitary = False)
full_double_circuit = simulation.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + injection_circuit + double_ghz_circuit + syndrome_circuit + double_ghz_circuit
full_ghz_circuit = simulation.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + injection_circuit + syndrome_circuit + syndrome_circuit + ghz_circuit + syndrome_circuit + ghz_circuit
full_double_circuit_correct = simulation.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + injection_circuit +syndrome_circuit + double_ghz_circuit  

rotated_simulation = RotatedSurfaceCodeSimulation(d = 3)
r_init_x_circuit = rotated_simulation.generate_init_circuit(basis = 'x')
r_injection_x_circuit = rotated_simulation.generate_injection_circuit_HXY(init_basis = 'x')
r_double_ghz_circuit_HXY = rotated_simulation.generate_projection_double_ghz_HXY()
r_syndrome_circuit = rotated_simulation.generate_syndrome_circuit()
r_full_double_circuit = r_init_x_circuit + r_injection_x_circuit + r_syndrome_circuit + r_double_ghz_circuit_HXY 

parallel = Parallel(n_jobs = -1)

#ps = [8e-3, 6e-3, 4e-3, 2e-3, 1e-3]
#ps = [4e-3, 2e-3, 1e-3]
ps = [1e-3] 
shots_per_batch = 18
total_shots = 10000
num_batches = total_shots // shots_per_batch

# Pre-compile all samplers
sampler = VecInterceptSampler(
        logical_x=[q.index for q in simulation.logical_x_qubits],
        logical_z=[q.index for q in simulation.logical_z_qubits])
    
rotated_sampler = VecInterceptSampler(
        logical_x=[q.index for q in rotated_simulation.logical_x_qubits],
        logical_z=[q.index for q in rotated_simulation.logical_z_qubits])

circ_to_sampler = {
    'rotated_HXY_injection_double_ghz': rotated_sampler,
    'unrotated_H_inj_double_correct': sampler,
}
circuits = {}
for p in ps:
    # for circ_name, circuit in [('unrotated_H_inj_double_correct', full_double_circuit_correct, 'H')]:
    for circ_name, circuit, measured_operator in [('rotated_HXY_injection_double_ghz', r_full_double_circuit, 'HXY')]:
        if p == 0:
            noise_model = None
        else:
            noise_model = NoiseModel.uniform_depolarizing(p=p)
        circuits[(p, circ_name, measured_operator)] = circuit.to_stim_circuit(noise_model, p)

def sample(p, circuit_name, measured_operator = 'H', shots = 1):
    circuit = circuits[(p, circuit_name, measured_operator)]
    task = sinter.Task(circuit=circuit)
    sampler = circ_to_sampler[circuit_name]
    compiled_sampler = sampler.compiled_sampler_for_task(task)
    return compiled_sampler.sample(shots = shots, measured_operator = measured_operator)


if __name__ == '__main__':
    try:    
        #output_file = f"/home/data/yotam/full_vec_simulation/yotam/data/{sys.argv[1]}"
        # output_file = f"/home/data/yotam/full_vec_simulation/double_correct_third_{sys.argv[1]}"
        output_file = f"/home/data/yotam/full_vec_simulation/rotated_HXY_injection_double_ghz_{sys.argv[1]}"
    except:
        raise ValueError('No output file provided')
    try:
        # Try to load existing results if file exists
        results = pd.read_csv(output_file).to_dict('records')
    except FileNotFoundError:
        # If file doesn't exist, start with empty list
        results = []
    for batch in (range(num_batches)):
        # Cycle through all ps in each batch
        for p, circuit_name, measured_operator in circuits.keys():
            t = time.time()
            batch_results = parallel(delayed(sample)(p, circuit_name, measured_operator) for _ in range(shots_per_batch))
            batch_results = sum(batch_results[1:], batch_results[0])  # Flatten results
            eval_time = time.time() - t
    
            # Ad metadata and save results
            results.append({
                'p': p,
                'batch': batch,
                'shots':  batch_results.shots,
                'discards' : batch_results.discards,
                'errors' : batch_results.errors,
                'circuit' : circuit_name,
                'eval_time' : eval_time
            })
    
            # Sve after each p's batch
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
