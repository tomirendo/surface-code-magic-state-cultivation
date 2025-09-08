from __future__ import annotations
import stim
import numpy as np
from pyperclip import copy
from tqdm import tqdm
import pymatching

from itertools import zip_longest
from joblib import Parallel, delayed
#%%
import sys; sys.path.insert(0, 'src/')
#%%
import gen
#%%
import pandas as pd
from plotly import express as px
import gen
import matplotlib.pyplot as plt
from json import loads


ps = np.array([2e-3, 1e-3, 5e-4, 3e-4, 2e-4, 1e-4])

if __name__ == '__main__':
    for p in ps:
                
        for noise, noise_name in zip([gen.NoiseModel.uniform_depolarizing(p), gen.NoiseModel.uniform_depolarizing_neutral_atoms(p)], ["uniform","uniform_atoms"]):
            
            circuit = stim.Circuit.from_file("noiseless-4q-code_circuit.stim")
            noisey_circuit = noise.noisy_circuit_skipping_mpp_boundaries(circuit)
            noisey_circuit.to_file(
                f"circuits/c=4q-init,d1=2,p={p},num_ghz_measurements=2,noise={noise_name}.stim")

            circuit = stim.Circuit.from_file("noiseless-4q-code-circuit-ghz-type.stim")
            noisey_circuit = noise.noisy_circuit_skipping_mpp_boundaries(circuit)
            noisey_circuit.to_file(
                f"circuits/c=4q-init-ghz-type,d1=2,p={p},num_ghz_measurements=2,noise={noise_name}.stim")

            circuit = stim.Circuit.from_file("noiseless-color-code-circuit-2ghz.stim")
            noisey_circuit = noise.noisy_circuit_skipping_mpp_boundaries(circuit)
            noisey_circuit.to_file(
                f"circuits/c=init-color-code,d1=3,p={p},num_ghz_measurements=2,noise={noise_name}.stim")

            circuit = stim.Circuit.from_file("noiseless-color-code-circuit-3ghz.stim")
            noisey_circuit = noise.noisy_circuit_skipping_mpp_boundaries(circuit)
            noisey_circuit.to_file(
            f"circuits/c=init-color-code,d1=3,p={p},num_ghz_measurements=3,noise={noise_name}.stim")

