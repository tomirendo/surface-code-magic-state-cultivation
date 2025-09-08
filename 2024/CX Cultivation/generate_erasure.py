import stim
import numpy as np
import os

def sample_erasure(circuit :  stim.Circuit, p : float, e : float) -> stim.Circuit:
    new_circuit = stim.Circuit()

    def analyze_line(new_circuit,line):
        if line.name == "DEPOLARIZE2":
            if p:
                new_circuit.append("DEPOLARIZE2",line.targets_copy(), p)
            target_pairs = list(zip(*[iter(line.targets_copy())]*2))
            erased = sum([pair for pair in target_pairs if np.random.rand() < e ],tuple())
            if erased:
                new_circuit.append("DEPOLARIZE2", 
                              erased,
                              15/16*.999999999)
        elif line.name == "DEPOLARIZE1":
            if p:
                new_circuit.append("DEPOLARIZE1",line.targets_copy(), p)
            targets = line.targets_copy()
            erased = [q for q in targets if np.random.rand() < e ]
            if erased:
                new_circuit.append("DEPOLARIZE1", 
                              erased,
                              3/4 * .999999999)
        else:
            new_circuit.append(line)


    for line in circuit:
            if line.name == 'REPEAT' :
                for _ in range(line.repeat_count):
                    for subline in line.body_copy():
                        analyze_line(new_circuit, subline)
            else:
                analyze_line(new_circuit, line)

    return new_circuit
ds = [3, 5, 7, 9, 11]
ps = [2e-2, 1e-2, 7e-3, 5e-3]
#ps = np.linspace(2e-2, 4e-3, 20)


for d in ds:
    for p in ps:
        filename = f"erasure_circuits/c=no-erasure,p={p},d={d}.stim"

        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=d,
            distance=d,
            after_clifford_depolarization=p)
        
        if not os.path.isfile(filename):
            circuit.to_file(filename)
            
        e = p 
        p_e = p / 2
            
        erasure_filename = f"erasure_circuits/c=erasure,p={p_e},e={e},d={d}.stim"
        erasure_circuit = sample_erasure(circuit, p=p_e, e=e)
        erasure_circuit.to_file(erasure_filename)
        
        e = p
        p_e = p / 10
            
        erasure_filename = f"erasure_circuits/c=erasure,p={p_e},e={e},d={d}.stim"
        erasure_circuit = sample_erasure(circuit, p=p_e, e=e)
        erasure_circuit.to_file(erasure_filename)
        
        e = p 
        p_e = p / 100
            
        erasure_filename = f"erasure_circuits/c=erasure,p={p_e},e={e},d={d}.stim"
        erasure_circuit = sample_erasure(circuit, p=p_e, e=e)
        erasure_circuit.to_file(erasure_filename)
            
        
        
