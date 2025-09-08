import sys
sys.path.append("../../2024/CX Cultivation/")
sys.path.append("../../2024/CX Cultivation/src")
sys.path.append("../StateVecSimulator/latte/")
sys.path.insert(0, ".")
from generate_d3_init import *
import stim
import numpy as np
from _noise import NoiseModel

import simulation
import importlib
from simulation import *
from IPython import display
import matplotlib.pyplot as plt
import stim
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm



if __name__ == "__main__":
    import os
    d1 = 3

    expanded_ps = [1e-3]
    d2s = [11]

    # Create circuits/CXCultivation directory if it doesn't exist
    os.makedirs("circuits/CXCultivation", exist_ok=True)
    def save_circ(p, num_ghz_measurements):
            stim.Circuit.cx = lambda self, targets: self.append("CX", [i.idx for i in targets])
            stim.Circuit.swap = lambda self, targets: self.append("SWAP", [i.idx for i in targets])
            stim.Circuit.h = lambda self, targets: self.append("H", [i.idx for i in targets])
            stim.Circuit.r = lambda self, targets: self.append("R", [i.idx for i in targets])
            stim.Circuit.rx = lambda self, targets: self.append("RX", [i.idx for i in targets])
            stim.Circuit.mr = lambda self, targets: self.append("MR", [i.idx for i in targets])
            stim.Circuit.mrx = lambda self, targets: self.append("MRX", [i.idx for i in targets])
            stim.Circuit.tick = lambda self: self.append("TICK")

            for transversal_cx in [True, False]:
                for noise, noise_name in zip([NoiseModel().uniform_depolarizing(p)], ["uniform"]):
                #for noise, noise_name in zip([NoiseModel.uniform_depolarizing(p), NoiseModel.uniform_depolarizing_neutral_atoms(p)], ["uniform","uniform_atoms"]):
                    circuit = generate_unexpanded(num_ghz_measurements, p = p, apply_transversal_cx = transversal_cx)
                    noisey_circuit = noise.noisy_circuit_skipping_mpp_boundaries(circuit)
                    noisey_circuit.to_file(
                    f"circuits/CXCultivation/c=init-cx,d1={d1},p={p},num_ghz_measurements={num_ghz_measurements},noise={noise_name},transversal_cx={transversal_cx}.stim")

    ps = np.array([2e-3, 1e-3, 5e-4, 3e-4, 2e-4, 1e-4])
    parallel(delayed(save_circ)(p, num_ghz_measurements)
             for p in tqdm(ps)
             for num_ghz_measurements in tqdm([1, 2, 3], leave=False))
    
    simulation = Simulation(d = d1)

    ghz_circuit = simulation.generate_projection_circuit(generate_unitary = False)
    double_ghz_projection_circuit = simulation.generate_double_ghz(generate_unitary = False)
    double_ghz_projection_circuit_HXY = simulation.generate_double_ghz_HXY(generate_unitary = False)
    syndrome_circuit = simulation.generate_syndrome_circuit(generate_unitary = False)
    injection_circuit = simulation.generate_injection_circuit(generate_unitary = False)
    injection_circuit_HXY = simulation.generate_injection_circuit_HXY(generate_unitary = False, init_basis='z')
    full_circuit = simulation.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + ghz_circuit + syndrome_circuit + ghz_circuit + syndrome_circuit + ghz_circuit
    short_circuit = simulation.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + ghz_circuit 
    mid_circuit = simulation.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + ghz_circuit + syndrome_circuit + ghz_circuit


    os.makedirs("circuits/HCultivationSurfaceCode", exist_ok=True)

    for num_ghz_measurements, circuit in zip([3,2,1], [full_circuit, mid_circuit, short_circuit]):
        for p in ps:
            for noise, noise_name in zip([NoiseModel().uniform_depolarizing(p)], ["uniform"]):
                noisey_circuit = circuit.to_stim_circuit(noise_model = noise, p = p, apply_non_cliffords = False)
                noisey_circuit.to_file(
                f"circuits/HCultivationSurfaceCode/c=init-H,d1={d1},p={p},num_ghz_measurements={num_ghz_measurements},noise={noise_name}.stim")


    double_ghz_circuit = simulation.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + simulation.generate_double_ghz(generate_unitary = False)
    double_ghz_circuit_measure_CZ = simulation.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + simulation.generate_double_ghz(generate_unitary = False)

    for p in ps:
        for noise, noise_name in zip([NoiseModel().uniform_depolarizing(p)], ["uniform"]):
            for circuit, circuit_name, non_clifford_noise_strategy in zip([double_ghz_circuit, double_ghz_circuit_measure_CZ], 
                                                                          ["double", "double-measure-CZ"], 
                                                                          ["NOISE", "CZ"]):
                noisey_circuit = circuit.to_stim_circuit(noise_model = noise, p = p, apply_non_cliffords = False, non_clifford_noise_strategy = non_clifford_noise_strategy)
                noisey_circuit.to_file(
                f"circuits/HCultivationSurfaceCode/c=init-H-{circuit_name},d1={d1},p={p},noise={noise_name}.stim")


    injection_syndrome_ghz_circuit = simulation.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + injection_circuit + syndrome_circuit + double_ghz_projection_circuit
    injection_ghz_circuit = simulation.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + injection_circuit + double_ghz_projection_circuit

    for p in list(ps) + [8e-3, 6e-3, 4e-3]:
        for circuit, circuit_name in zip([injection_syndrome_ghz_circuit, injection_ghz_circuit], ["injection-syndrome-ghz", "injection-ghz"]):
            for noise, noise_name in zip([NoiseModel().uniform_depolarizing(p)], ["uniform"]):
                noisey_circuit = circuit.to_stim_circuit(noise_model = noise, p = p, apply_non_cliffords = False, non_clifford_noise_strategy = "CZ")
                noisey_circuit.to_file(
                f"circuits/HCultivationSurfaceCode/c=init-H-{circuit_name},d1={d1},p={p},noise={noise_name}.stim")

    injection_syndrome_ghz_circuit_HXY  = simulation.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + injection_circuit_HXY + syndrome_circuit + double_ghz_projection_circuit_HXY
    injection_ghz_circuit_HXY = simulation.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + injection_circuit_HXY + double_ghz_projection_circuit_HXY
    for p in ps:
        for noise, noise_name in zip([NoiseModel().uniform_depolarizing(p)], ["uniform"]):
            for circuit, circuit_name in zip([injection_syndrome_ghz_circuit_HXY, injection_ghz_circuit_HXY], ["injection-syndrome-ghz-HXY", "injection-ghz-HXY"]):
                noisey_circuit = circuit.to_stim_circuit(noise_model = noise, p = p, apply_non_cliffords = False, non_clifford_noise_strategy = "CZ_HXY")
                noisey_circuit.to_file(
                f"circuits/HCultivationSurfaceCode/c=init-HXY-{circuit_name},d1={d1},p={p},noise={noise_name}.stim")



    
    simulation_rotated = RotatedSurfaceCodeSimulation(d = 3)
    rotated_ghz_circuit_HXY = simulation_rotated.generate_projection_circuit_HXY()
    rotated_syndrome_circuit = simulation_rotated.generate_syndrome_circuit()
    rotated_injection_circuit_HXY = simulation_rotated.generate_injection_circuit_HXY(init_basis = 'z')
    rotated_double_ghz_circuit_HXY = simulation_rotated.generate_projection_double_ghz_HXY()

    rotated_injection_ghz_circuit_HXY = simulation_rotated.generate_syndrome_circuit(detect_x = False) + rotated_injection_circuit_HXY + rotated_ghz_circuit_HXY + rotated_ghz_circuit_HXY
    rotated_injection_double_ghz_circuit_HXY = simulation_rotated.generate_syndrome_circuit(detect_x = False) + rotated_injection_circuit_HXY + rotated_double_ghz_circuit_HXY
    rotated_injection_syndrome_double_ghz_circuit_HXY = simulation_rotated.generate_syndrome_circuit(detect_x = False) + rotated_injection_circuit_HXY + rotated_syndrome_circuit + rotated_double_ghz_circuit_HXY 

    rotated_init_x_circuit = simulation_rotated.generate_init_circuit(basis = 'x')
    rotated_injection_circuit_HXY_init_x = simulation_rotated.generate_injection_circuit_HXY(init_basis = 'x')

    rotated_init_x_injection_double_HXY = rotated_init_x_circuit + rotated_injection_circuit_HXY_init_x + rotated_double_ghz_circuit_HXY
    rotated_init_x_injection_syndrome_double_ghz_circuit_HXY = rotated_init_x_circuit + rotated_injection_circuit_HXY_init_x + rotated_syndrome_circuit + rotated_double_ghz_circuit_HXY


    for p in list(ps) + [8e-3, 6e-3, 4e-3]:
        for noise, noise_name in zip([NoiseModel().uniform_depolarizing(p)], ["uniform"]):
            for circuit, circuit_name, non_clifford_def in zip([rotated_injection_ghz_circuit_HXY, 
                                              rotated_injection_double_ghz_circuit_HXY, 
                                              rotated_injection_syndrome_double_ghz_circuit_HXY,
                                              rotated_init_x_injection_double_HXY,
                                              rotated_init_x_injection_syndrome_double_ghz_circuit_HXY], 
                                             ["rotated-injection-ghz-HXY", 
                                              "rotated-injection-double-ghz-HXY", 
                                              "rotated-injection-syndrome-double-ghz-HXY",
                                              "rotated-init-x-injection-double-HXY",
                                              "rotated-init-x-injection-syndrome-double-ghz-HXY"],
                                             ['CZ_HXY']*3+['CX_HXY']*2):
                measurement_basis = 'z' if non_clifford_def == 'CZ_HXY' else 'x'
                noisey_circuit = circuit.to_stim_circuit(noise_model = noise, p = p, apply_non_cliffords = False, non_clifford_noise_strategy = non_clifford_def, logical_measurement_basis = measurement_basis)
                noisey_circuit.to_file(
                    f"circuits/HCultivationSurfaceCode/c=init-HXY-{circuit_name},d1={d1},p={p},noise={noise_name}.stim")

    for p in expanded_ps:
        for d2 in d2s:
            for noise, noise_name in zip([NoiseModel().uniform_depolarizing(p), NoiseModel().uniform_depolarizing_neutral_atoms(p)], ["uniform", "uniform_atoms"]):
                for circuit, circuit_name, non_clifford_def in zip(
                    [rotated_injection_double_ghz_circuit_HXY,
                    rotated_injection_syndrome_double_ghz_circuit_HXY,
                    rotated_init_x_injection_double_HXY],
                    ['expanded-rotated-injection-double-HXY', 
                     'expanded-rotated-syndrome-injection-double-HXY',
                     'expanded-rotated-init-x-injection-double-HXY'],
                    ['CZ_HXY','CZ_HXY','CX_HXY']):
                    measurement_basis = 'z' if non_clifford_def == 'CZ_HXY' else 'x'
                    expanded_circuit = circuit.to_stim_circuit(
                        post_select_syndromes = False, 
                        measure_logical_operator = False, 
                        apply_non_cliffords = False, p = p,
                        non_clifford_noise_strategy = non_clifford_def,
                        logical_measurement_basis = measurement_basis) + \
                        simulation_rotated.stim_expansion_circuit(d2 = d2, rounds = d2, logical_measurement_basis = measurement_basis)
                    noisey_circuit = noise.noisy_circuit_skipping_mpp_boundaries(expanded_circuit)
                    noisey_circuit.to_file(
                f"circuits/HCultivationSurfaceCode/c={circuit_name},d2={d2},d1={d1},p={p},noise={noise_name}.stim")

            for noise, noise_name in zip([NoiseModel().uniform_depolarizing(p), NoiseModel().uniform_depolarizing_neutral_atoms(p)], ["uniform", "uniform_atoms"]):
                for circuit, circuit_name, non_clifford_def in zip(
                        [injection_ghz_circuit,injection_syndrome_ghz_circuit],
                        ['expanded-injection-ghz','expanded-injection-syndrome-ghz'],
                        ['CZ','CZ']):
                    measurement_basis = 'z' if non_clifford_def in ['CZ','CZ_HXY'] else 'x'
                    expanded_circuit = circuit.to_stim_circuit(
                        post_select_syndromes = False, 
                        measure_logical_operator = False, 
                        apply_non_cliffords = False, p = p,
                        non_clifford_noise_strategy = non_clifford_def,
                        logical_measurement_basis = measurement_basis) + \
                        simulation.stim_expansion_circuit(d2 = d2, rounds = d2, logical_measurement_basis = measurement_basis)
                    noisey_circuit = noise.noisy_circuit_skipping_mpp_boundaries(expanded_circuit)
                    noisey_circuit.to_file(
                f"circuits/HCultivationSurfaceCode/c={circuit_name},d2={d2},d1={d1},p={p},noise={noise_name}.stim")


    # # simulation_d5 = Simulation(d = 5)

    # # double_ghz_d5 = simulation_d5.generate_double_ghz(generate_unitary = False)
    # # syndrome_d5 = simulation_d5.generate_syndrome_circuit(generate_unitary = False)
    # # injection_d5 = simulation_d5.generate_injection_circuit(generate_unitary = False)

    # # injection_d5 = simulation_d5.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + injection_d5 + double_ghz_d5 + double_ghz_d5
    # # injection_syndrome_d5 = simulation_d5.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + injection_d5 + syndrome_d5 + double_ghz_d5 + double_ghz_d5

    # # for p in ps:
    # #     for noise, noise_name in zip([NoiseModel().uniform_depolarizing(p)], ["uniform"]):
    # #         for circuit, circuit_name in zip([injection_syndrome_d5, injection_d5], 
    # #                                          ["injection-syndrome-ghz", "injection-ghz"]):
    # #             noisey_circuit = circuit.to_stim_circuit(noise_model = noise, p = p, apply_non_cliffords = False, non_clifford_noise_strategy = "CZ")
    # #             noisey_circuit.to_file(
    # #             f"circuits/HCultivationSurfaceCode/c=init-{circuit_name},d1={5},p={p},noise={noise_name}.stim")
    

    simulation = RotatedSurfaceCodeSimulation(d = 3)
    init_x_circuit = simulation.generate_init_circuit(basis = 'x')
    injection_HXY_init_x = simulation.generate_injection_circuit_HXY(init_basis = 'x')
    double_ghz_HXY = simulation.generate_projection_double_ghz_HXY()
    double_ghz_CZ = simulation.generate_projection_double_ghz_HXY(replace_phase_kickback = True, target_gate = CX)

    d3_circuit = init_x_circuit + injection_HXY_init_x + double_ghz_CZ
    d5_simulation, d3_circuit_expanded = d3_circuit.expand_simulation(new_distance = 5)
    expansion_circuit = d5_simulation.circuit_from_stim_file("./circuits/CrumbleEdits/rotated_collaps_d5_to_d3.stim",reverse = True)
    double_ghz_d5 = d5_simulation.generate_projection_double_ghz_HXY()
    d5_measure_syndromes = d5_simulation.generate_syndrome_circuit()
    d5_circuit = expansion_circuit + double_ghz_d5
    d5_circuit_with_syndromes = expansion_circuit + d5_measure_syndromes + double_ghz_d5

    for p in ps:
        for noise, noise_name in zip([NoiseModel().uniform_depolarizing(p), NoiseModel().uniform_depolarizing_neutral_atoms(p)], ["uniform", "uniform_atoms"]):
            small_circuit = d3_circuit_expanded.to_stim_circuit(noise_model = noise, p = p, apply_non_cliffords = False, non_clifford_noise_strategy = 'CZ_HXY', measure_logical_operator=False,post_select_syndromes=False)
            larger_circuit = d5_circuit.to_stim_circuit(noise_model = noise, p = p, apply_non_cliffords = False, non_clifford_noise_strategy = 'CX_HXY', logical_measurement_basis = 'x')
            full_circuit = small_circuit + larger_circuit
            full_circuit.to_file(
                f"circuits/HCultivationSurfaceCode/c=expansion-rotated-HXY-d3-d5,d1={3},d2={5},p={p},noise={noise_name}.stim")
            
            larger_circuit_with_syndromes = d5_circuit_with_syndromes.to_stim_circuit(noise_model = noise, p = p, apply_non_cliffords = False, non_clifford_noise_strategy = 'CX_HXY', logical_measurement_basis = 'x')
            full_circuit_with_syndromes = small_circuit + larger_circuit_with_syndromes
            full_circuit_with_syndromes.to_file(
                f"circuits/HCultivationSurfaceCode/c=expansion-rotated-HXY-d3-d5-with-syndromes,d1={3},d2={5},p={p},noise={noise_name}.stim")
            
    for p in expanded_ps:
        for d2 in d2s:
            for noise, noise_name in zip([NoiseModel().uniform_depolarizing(p), NoiseModel().uniform_depolarizing_neutral_atoms(p)], ["uniform", "uniform_atoms"]):
                small_circuit = d3_circuit_expanded.to_stim_circuit(p = p, apply_non_cliffords = False, non_clifford_noise_strategy = 'CZ_HXY', measure_logical_operator=False,post_select_syndromes=False)
                larger_circuit_with_syndromes = d5_circuit_with_syndromes.to_stim_circuit(p = p, apply_non_cliffords = False, non_clifford_noise_strategy = 'CX_HXY', logical_measurement_basis = 'x', post_select_syndromes=False, measure_logical_operator=False)
                expansion_circuit = d5_simulation.stim_expansion_circuit(d2 = d2, rounds = d2,logical_measurement_basis = 'x')
                full_circuit = noise.noisy_circuit_skipping_mpp_boundaries(small_circuit + larger_circuit_with_syndromes + expansion_circuit)
                full_circuit.to_file(
                    f"circuits/HCultivationSurfaceCode/c=end2end-rotated-HXY-d3-d5-with-syndromes,d1={3},d2={5},d3={d2},p={p},noise={noise_name}.stim")

    for p_multiplier in [1, 2, 3, 6, 8, 12, 16]:
        p = 1e-3
        for noise, noise_name in zip([NoiseModel().uniform_depolarizing(p), NoiseModel().uniform_depolarizing_neutral_atoms(p)], ["uniform", "uniform_atoms"]):
            small_circuit = d3_circuit_expanded.to_stim_circuit(p = p/3*p_multiplier, apply_non_cliffords = False, non_clifford_noise_strategy = 'CZ_HXY', measure_logical_operator=False,post_select_syndromes=False)
            larger_circuit_with_syndromes = d5_circuit_with_syndromes.to_stim_circuit(p = p/3*p_multiplier, apply_non_cliffords = False, non_clifford_noise_strategy = 'CX_HXY', logical_measurement_basis = 'x', post_select_syndromes=False, measure_logical_operator=False)
            expansion_circuit = d5_simulation.stim_expansion_circuit(d2 = 9, rounds = 9,logical_measurement_basis = 'x')
            full_circuit = noise.noisy_circuit_skipping_mpp_boundaries(small_circuit + larger_circuit_with_syndromes + expansion_circuit)
            full_circuit.to_file(
                f"circuits/HCultivationSurfaceCode/c=compiled-3q-multiplier-{p_multiplier}-rotated-HXY,d1={3},d2={5},d3={9},p={1e-3*p_multiplier},noise={noise_name}.stim")

    
    simulation = Simulation(d = 2)
    ghz_circuit = simulation.generate_projection_circuit(generate_unitary = False)
    syndrome_circuit = simulation.generate_syndrome_circuit(generate_unitary = False)
    double_ghz_circuit = simulation.generate_double_ghz(generate_unitary = False)
    injection_circuit = simulation.generate_injection_circuit(generate_unitary = False)
    full_circuit = simulation.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + injection_circuit + ghz_circuit 
    full_circuit_injection_ghz = simulation.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + injection_circuit + syndrome_circuit + ghz_circuit 
    double_injection_ghz_circuit =  simulation.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + injection_circuit + double_ghz_circuit 
    double_injection_syndrome_ghz_circuit =  simulation.generate_syndrome_circuit(generate_unitary = False, detect_x = False) + injection_circuit + double_ghz_circuit + syndrome_circuit 


    for p in ps:
        for noise, noise_name in zip([NoiseModel().uniform_depolarizing(p), NoiseModel().uniform_depolarizing_neutral_atoms(p)], ["uniform", "uniform_atoms"]):
            for circuit, circuit_name in zip([full_circuit, full_circuit_injection_ghz, double_injection_ghz_circuit, double_injection_syndrome_ghz_circuit], ["injection-ghz", "injection-ghz-syndrome", "double-injection-ghz", "double-injection-ghz-syndrome"]):
                noisey_circuit = circuit.to_stim_circuit(noise_model = noise, p = p, apply_non_cliffords = False, non_clifford_noise_strategy = "CZ", logical_measurement_basis='z')
                noisey_circuit.to_file(
                f"circuits/HCultivationSurfaceCode/c=init-H-{circuit_name},d1={2},p={p},noise={noise_name}.stim")

    for p in ps:
        for d2 in d2s:
            for noise, noise_name in zip([NoiseModel().uniform_depolarizing(p), NoiseModel().uniform_depolarizing_neutral_atoms(p)], ["uniform", "uniform_atoms"]):
                for circuit, circuit_name, non_clifford_def in zip([full_circuit, full_circuit_injection_ghz, double_injection_ghz_circuit, double_injection_syndrome_ghz_circuit], ["injection-ghz", "injection-ghz-syndrome", "double-injection-ghz", "double-injection-ghz-syndrome"], ["CZ", "CZ", "CZ", "CZ"]):
                    expanded_circuit = circuit.to_stim_circuit(p = p, apply_non_cliffords = False, non_clifford_noise_strategy = non_clifford_def, measure_logical_operator=False,post_select_syndromes=False) + \
                        simulation.stim_expansion_circuit(d2 = d2, rounds = d2, logical_measurement_basis = 'z')
                    noisey_circuit = noise.noisy_circuit_skipping_mpp_boundaries(expanded_circuit)
                    noisey_circuit.to_file(
                f"circuits/HCultivationSurfaceCode/c=expansion-H-{circuit_name},d1={2},d2={d2},p={p},noise={noise_name}.stim")

            
 


           
