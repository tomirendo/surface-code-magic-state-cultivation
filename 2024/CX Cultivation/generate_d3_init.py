from __future__ import annotations
import stim
import numpy as np
from pyperclip import copy
from tqdm import tqdm
import pymatching

from itertools import zip_longest, product
from joblib import Parallel, delayed
#%%
import sys; sys.path.insert(0, 'src/')
#%%
try:
    import gen
except:
    pass
#%%
import pandas as pd
from plotly import express as px
import gen
import matplotlib.pyplot as plt
from json import loads

def get_d(json_metadata):
    return loads(json_metadata)['d']
def filter_generator(json_metadata, circuit_type='unitary'):
    return loads(json_metadata)['style'] == 'unitary'
#%%
stim.Circuit.cx = lambda self, targets : self.append("CX",[i.idx for i in targets])
stim.Circuit.swap = lambda self, targets : self.append("SWAP",[i.idx for i in targets])
stim.Circuit.h = lambda self, targets : self.append("H",[i.idx for i in targets])
stim.Circuit.r = lambda self, targets : self.append("R",[i.idx for i in targets])
stim.Circuit.rx = lambda self, targets : self.append("RX",[i.idx for i in targets])
stim.Circuit.mr = lambda self, targets : self.append("MR",[i.idx for i in targets])
stim.Circuit.mrx = lambda self, targets : self.append("MRX",[i.idx for i in targets])
stim.Circuit.tick = lambda self : self.append("TICK")
#%%

triple_qubit_error_count = 4**3 - 1

def _get_cnot_pairs(line):
    return list(zip(*[iter(line[3:].split(" "))]*2))

def add_error(bare_circuit, p=1e-3) -> list[stim.Circuit]:
    bare_circuit_lines = str(bare_circuit).split('\n')
    circuit = ""
    for line in bare_circuit_lines:
        circuit += line + "\n"
        if line.startswith("CX") or line.startswith('CZ') :
            cnot_pairs = _get_cnot_pairs(line)
            circuit += f"DEPOLARIZE2({p}) {line[:3]}\n"
    return stim.Circuit(circuit)


def get_qubit(x, y):
    return find_qubit([x, y], [0, 0], all_non_expanded_qubits)


def find_qubit(origin_qubit, shift, superset):
    for qubit in superset:
        if (np.array(list(origin_qubit)) + np.array(shift) == np.array(list(qubit))).all():
            return qubit
    
z_stabilizer_order = [(0.5,0.5), ( -0.5, 0.5), ( 0.5, -0.5), (-0.5, -0.5)]
x_stabilizer_order = [(0.5,0.5), ( 0.5, -0.5), (-0.5,  0.5), (-0.5, -0.5)]

class Qubit:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        global global_index
        self.idx = global_index
        global_index += 1
    
    def __repr__(self):
        if self.is_data():
            return f"Data({self.x}, {self.y} | {self.idx})"
        elif self.ancilla_type() == 'X':
            return f"X({self.x}, {self.y} | {self.idx})"
        elif self.ancilla_type() == 'Z':
            return f"Z({self.x}, {self.y} | {self.idx})"
    
    def coords(self):
        return list(self) + [0]
    
    @property
    def radius(self):
        if -0.5 in [self.x, self.x+horizontal_offset, self.y]:
            return 0
        
        return max(np.abs([self.x%horizontal_offset, self.y]))
    
    def is_data(self):
        return (self.x - int(self.x)) == 0.0
    
    def ancilla_type(self):
        if self.is_data():
            return 'D'
        if int(self.x +self.y ) % 2 == 0:
            return 'X'
        else:
            return 'Z'  
        
    def get_surface_code_cx(self, idx, superset):
        if self.is_data():
            raise Exception(f"Can't find CX from a data qubit {self}")
        if self.ancilla_type() == 'X':
            direction = x_stabilizer_order[idx]
        elif self.ancilla_type() == 'Z':
            direction = z_stabilizer_order[idx]
            
        companion_qubit = find_qubit(self, direction, superset)
        if companion_qubit is not None:
            if self.ancilla_type() == 'X':
                return [self, companion_qubit]
            elif self.ancilla_type() == 'Z':
                return [companion_qubit, self]
            else:
                raise Exception("Unrecognised ancilla type")
        else:
            return []
        
    def get_stabilizer_support(self, superset) -> list[Qubit]:
        return [q for q in sum((self.get_surface_code_cx(i, superset) 
                                for i in range(4)),[]) if q.is_data()]
    
    def search_single_qubit_fix(self, all_syndromes, superset):
        support = self.get_stabilizer_support(superset)
        for q in support:
            for syndrom in all_syndromes:
                if syndrom != self:
                    if q in syndrom.get_stabilizer_support(superset):
                        break
            else:
                return q
            
        raise Exception("Can't find single qubit fix, is d>3 ?")
    
    def correction_steps(self, step_index, superset):
        if self.x != horizontal_offset - 0.5:
            x = (self.x %  (horizontal_offset))
        else:
            x = -0.5
    
        y = self.y
        
        direction_x = 0.5 if (x > d//2) else -0.5
        direction_y = 0.5 if (y > d//2) else -0.5
        
        move_x = direction_x if ( 0 <= x + direction_x <= d-1) else -direction_x
        move_y = direction_y if ( 0 <= y + direction_y <= d-1) else -direction_y
        
        step = [move_x, move_y]
        for _ in range(step_index):
            if self.ancilla_type() == 'Z':
                step[0] += np.sign(direction_x)
            else:
                step[1] += np.sign(direction_y)  
        
        corr_qubit = find_qubit(self, step, superset)
        if corr_qubit is None:
            pass
        return corr_qubit
    
    def support_contained(self, containing_set, superset):
        for q in self.get_stabilizer_support(superset):
            if q not in containing_set:
                return False
        return True
        
    def support_intersect(self, intersection_set, superset):
        for q in self.get_stabilizer_support(superset):
            if q in intersection_set:
                return True
        return False
    
    def __iter__(self):
        return iter([self.x, self.y])
    
    def __ge__(self, other):
        return list(self).__ge__(list(other))
    
    def __lt__(self, other):
        return list(self).__lt__(list(other))

#%%
qubit_locations = [] 
ancilla_x_locations = [] 
ancilla_z_locations = [] 
data_locations = [] 
global_index = 0 

d = 3
d2 = 15
horizontal_offset = d2 + 1
assert d % 2 == 1

#print(f'd = {d}, d2 = {d2}')
surface_codes = []

for surface_index in [0,1,2]:
    surface_codes.append([])
    offset = horizontal_offset * surface_index 
    for x in range(d):
        for y in range (d):
            location = Qubit(x + offset, y )
            qubit_locations.append(location)
            surface_codes[-1].append(location)
            
            if ((y%2 == 0) and ((y != 0) or (x % 2 != 0))) and surface_index < 2:
                location = Qubit(x+0.5 + offset, y-0.5)
                qubit_locations.append(location)
                surface_codes[-1].append(location)
                
            if (y%2 == 1)  and surface_index < 2:
                location = Qubit(x-0.5 + offset, y-0.5)
                qubit_locations.append(location)
                surface_codes[-1].append(location)
                
        if x % 2 == 0 and x < d - 1 and surface_index < 2:
            location = Qubit(x+0.5 + offset, y+0.5)
            qubit_locations.append(location)
            surface_codes[-1].append(location)

def new_qubit_at_location(x, y):
    location = Qubit(x, y )
    if find_qubit(location, [0, 0], qubit_locations) is None:
        qubit_locations.append(location)
        return location
    else:
        return find_qubit(location, [0, 0], qubit_locations)
    
expanded_surface_codes = []

for surface_index in [0,1]:
    expanded_surface_codes.append([])
    offset = horizontal_offset * surface_index 
    for x in range(d2):
        for y in range (d2):
            location = new_qubit_at_location(x + offset, y)
            expanded_surface_codes[-1].append(location)
            if (y%2 == 0) and ((y != 0) or (x % 2 != 0)):
                location = new_qubit_at_location(x+0.5 + offset, y-0.5)
                expanded_surface_codes[-1].append(location)
            if (y%2 == 1) :
                location = new_qubit_at_location(x-0.5 + offset, y-0.5)
                expanded_surface_codes[-1].append(location)
        if x % 2 == 0 and x < d2 -1:
            location = new_qubit_at_location(x+0.5 + offset, y+0.5)
            expanded_surface_codes[-1].append(location)
#%%
ancilla_x_qubits : list[list[Qubit]] = []
ancilla_z_qubits : list[list[Qubit]] = []
data_qubits : list[list[Qubit]] = []

for idx, surface_code in enumerate(surface_codes[:2]):
    ancilla_x_qubits.append([])
    ancilla_z_qubits.append([])
    data_qubits.append([])
    
    for qubit in surface_code:
        if qubit.ancilla_type() == 'X':
            ancilla_x_qubits[idx].append(qubit)
        elif qubit.ancilla_type() == 'Z':
            ancilla_z_qubits[idx].append(qubit)
        elif qubit.ancilla_type() == 'D':
            data_qubits[idx].append(qubit)
            
ghz_qubits : list[Qubit] = [q for q in surface_codes[2] if q.ancilla_type() == 'D'] 
if d == 3:
    mid_ghz_qubit : list[Qubit] = find_qubit(Qubit(int(horizontal_offset * 2 + d//2), d//2), 
                                         [0,0], qubit_locations)
elif d == 5:
    mid_ghz_qubit : list[Qubit] = find_qubit(Qubit(int(horizontal_offset * 2 + d//2), d//2),
                                             [-1,0], qubit_locations)
else:
    raise NotImplementedError('No ghz state for d > 5')

non_mid_ghz_qubit : list[Qubit] = [q for q in ghz_qubits if q != mid_ghz_qubit]

ancilla_measurement_order : list[Qubit] = sum(ancilla_z_qubits+ ancilla_x_qubits,[])

ghz_meas_offset = len(ghz_qubits)
x_init_data : list[Qubit] = []
z_init_data : list[Qubit] = []

for index, code in enumerate(expanded_surface_codes):
    x_init_data += [q for q in code if q.ancilla_type() == 'D'
                    and (q.x - offset*index) - q.y >= 0 
                    and ((q.x - offset*index) >= d or q.y >= d)]    
    z_init_data += [q for q in code if q.ancilla_type() == 'D'
                    and (q.x - offset*index) - q.y < 0 
                    and ((q.x - offset*index) >= d or q.y >= d)]
        
z_ancillas_expanded : list[Qubit] = [q for q in sum(expanded_surface_codes,[]) if q.ancilla_type() == 'Z']
x_ancillas_expanded : list[Qubit] = [q for q in sum(expanded_surface_codes,[]) if q.ancilla_type() == 'X']
expanded_data_qubits : list[Qubit] = [q for q in sum(expanded_surface_codes,[]) if q.ancilla_type() == 'D']

measurement_order_expanded : list[Qubit] = x_ancillas_expanded+z_ancillas_expanded

all_non_expanded_qubits = sum(surface_codes,[])
all_expanded_qubits = sum(expanded_surface_codes,[])
#%%
#%%
def init_circuit(circuit, gidney_style = False, random_init = True):
    for qubit in qubit_locations:
        circuit.append("QUBIT_COORDS",qubit.idx, list(qubit))
        
    circuit.rx(data_qubits[0])
    circuit.r(data_qubits[1])
     
    """
    Initilization for CZ magic state distillation
    circuit.rx(data_qubits[0])
    circuit.r(data_qubits[1])
    """
    
    for x_syn in ancilla_x_qubits[:2]:
        circuit.rx(x_syn)
    for z_syn in ancilla_z_qubits[:2]:
        circuit.r(z_syn)
        
   
    circuit.tick()
    circuit = stabilizer_cycle(circuit)
    
    for step_index in range(int(d // 2)):
    
        for z_syn in ancilla_z_qubits[0]:
            meas_index = ancilla_measurement_order.index(z_syn)
            correction_qubit = z_syn.correction_steps(step_index, all_non_expanded_qubits)
            if correction_qubit is not None:
                circuit.append("CX", [stim.target_rec(meas_index - len(ancilla_measurement_order)), correction_qubit.idx])

        for x_syn  in ancilla_x_qubits[1]:
            meas_index = ancilla_measurement_order.index(x_syn)
            correction_qubit = x_syn.correction_steps(step_index, all_non_expanded_qubits)
            if correction_qubit is not None:
                circuit.append("CZ", [stim.target_rec(meas_index - len(ancilla_measurement_order)), correction_qubit.idx])
        circuit.tick()
        
    circuit.cx(
        sum([[qubit, find_qubit(qubit, [horizontal_offset, 0], all_non_expanded_qubits)] 
             for qubit in surface_codes[0] if qubit.ancilla_type() == 'D'], [])
    )
    if not gidney_style:
        circuit.r(non_mid_ghz_qubit)
        circuit.rx([mid_ghz_qubit])
    else:
        circuit.rx(non_mid_ghz_qubit + [mid_ghz_qubit])
        
    
    circuit.tick()
    
    if random_init:
        circuit.append("CORRELATED_ERROR",
                   [stim.target_pauli(qubit.idx, "Z")
                    for qubit in surface_codes[0]
                    if qubit.ancilla_type() == 'D'
                    ], 1/4)

    return circuit

#%%
def stabilizer_cycle(circuit : stim.Circuit) -> stim.Circuit:
    all_ancillas = sum(ancilla_x_qubits + ancilla_z_qubits,[])
    cycles = [
        sum([qubit.get_surface_code_cx(idx, all_non_expanded_qubits) for qubit in all_ancillas], [])
        for idx in range(4)
    ]
    for cycle in cycles:
        circuit.cx(cycle)
        circuit.tick()
    circuit.mr([q for q in ancilla_measurement_order if q.ancilla_type() == 'Z'])
    circuit.mrx([q for q in ancilla_measurement_order if q.ancilla_type() == 'X'])
    circuit.tick()
    return circuit
#%%

def stabilizer_ghz_cycle(circuit : stim.Circuit, p = None, apply_transversal_cx = True) -> stim.Circuit:
    all_ancillas = sum(ancilla_x_qubits + ancilla_z_qubits,[])
    ghz_cycles = generate_ghz_cycles()
    inverse_ghz_cycle = ghz_cycles[::-1]
    cycles = [
            sum([qubit.get_surface_code_cx(idx, all_non_expanded_qubits) for qubit in all_ancillas], [])
            for idx in range(4)
    ]
        
    for cycle,ghz_cycle in zip_longest(cycles, ghz_cycles):
        if cycle is not None:
            circuit.cx(cycle + ghz_cycle)
        else:
            circuit.cx(ghz_cycle)
        circuit.tick()
            
    circuit.mr([q for q in ancilla_measurement_order if q.ancilla_type() == 'Z'])
    circuit.mrx([q for q in ancilla_measurement_order if q.ancilla_type() == 'X'])
    circuit.tick()
    
    for idx in range(len(all_ancillas)):
        circuit.append("DETECTOR",stim.target_rec(-1-idx))

    if apply_transversal_cx:
        transversal_cx(circuit, d=d, p=p)
    else:
        transversal_error(circuit, d=d, p=p)
        two_qubit_transversal_error(circuit, d=d, p=p)
    
    for cycle,ghz_cycle in zip_longest(cycles, inverse_ghz_cycle):
        if cycle is not None:
            circuit.cx(cycle + ghz_cycle)
        else:
            circuit.cx(ghz_cycle)
        circuit.tick()
    
    circuit.mr([q for q in ancilla_measurement_order if q.ancilla_type() == 'Z'] + non_mid_ghz_qubit)
    circuit.mrx([q for q in ancilla_measurement_order if q.ancilla_type() == 'X'] + [mid_ghz_qubit])
    circuit.tick()
    
    for idx in range(len(all_ancillas) + len(ghz_qubits)):
        circuit.append("DETECTOR",stim.target_rec(-1-idx))
    
    return circuit

#%%
def transversal_cx(circuit : stim.Circuit, d : int, inverted_order = False, p = None) -> stim.Circuit:
    if p is None:
        raise Exception("Transversal CX called without parameter p")
    if inverted_order:
        step_order = -1
    else:
        step_order = 1
    circuit.cx(
        sum([[qubit, find_qubit(qubit, [-(2 * horizontal_offset), 0], all_non_expanded_qubits)] for qubit in ghz_qubits], [])[::step_order]
    )
    circuit.tick()
    
    circuit.cx(
        sum([[qubit, find_qubit(qubit, [-(horizontal_offset), 0], all_non_expanded_qubits)] for qubit in ghz_qubits], [])[::step_order]
    )
    transversal_error(circuit, d=d, p=p)
    circuit.tick()
    return circuit
    
def transversal_error(circuit : stim.Circuit, d : int, p : float) -> stim.Circuit:
    for qubit in ghz_qubits:
        q1 = find_qubit(qubit, [-(2*horizontal_offset), 0], all_non_expanded_qubits)
        q2 = find_qubit(qubit, [-(horizontal_offset), 0], all_non_expanded_qubits)
        for paulis in product(*["IXYZ"]*3):
            pauli_noise = [stim.target_pauli(q.idx, p)
                            for p,q in zip(paulis, [q1,q2,qubit])
                                if p != 'I']
            if pauli_noise:
                circuit.append("CORRELATED_ERROR",pauli_noise, 3*p/triple_qubit_error_count)
    return circuit
def two_qubit_transversal_error(circuit : stim.Circuit, d : int, p : float) -> stim.Circuit:
    for qubit in ghz_qubits:
        q1 = find_qubit(qubit, [-(2*horizontal_offset), 0], all_non_expanded_qubits)
        q2 = find_qubit(qubit, [-(horizontal_offset), 0], all_non_expanded_qubits)
        circuit.append("DEPOLARIZE2", [qubit.idx, q1.idx], p)
        circuit.append("DEPOLARIZE2", [qubit.idx, q2.idx], p)
    return circuit
#%%
def measure_XX_expanded(circuit : stim.Circuit) -> stim.Circuit:
    circuit.append("MPP",
                   stim.target_combined_paulis([stim.target_pauli(qubit.idx, "X")
                    for qubit in expanded_data_qubits
                    if qubit.y == 0 
                    ]))
    circuit.append("OBSERVABLE_INCLUDE",stim.target_rec(-1), 0)
    return circuit
    
def measure_ZZ_expanded(circuit : stim.Circuit) -> stim.Circuit:
    circuit.append("MPP",
                   stim.target_combined_paulis([stim.target_pauli(qubit.idx, "Z")
                    for qubit in expanded_data_qubits
                    if qubit.x%horizontal_offset == 0
                    ]))
    circuit.append("OBSERVABLE_INCLUDE",stim.target_rec(-1), 1)
    return circuit
    
    
def measure_all_stabs(circuit : stim.Circuit) -> stim.Circuit:
    for syn in measurement_order_expanded:
        circuit.append("MPP",
                   stim.target_combined_paulis([stim.target_pauli(qubit.idx, syn.ancilla_type())
                    for qubit in syn.get_stabilizer_support(all_expanded_qubits)
                    ]))
    for idx, syn in enumerate(measurement_order_expanded):
        circuit.append("DETECTOR",[stim.target_rec(idx - len(measurement_order_expanded)),
                                   stim.target_rec(idx - len(measurement_order_expanded)*2)], [1000, 1000])
    return circuit
        
#%%
def measure_XX(circuit : stim.Circuit) -> stim.Circuit:
    circuit.append("MPP",
                   stim.target_combined_paulis([stim.target_pauli(qubit.idx, "X")
                    for qubit in surface_codes[0] + surface_codes[1] 
                    if qubit.ancilla_type() == 'D'
                    ]))
    circuit.append("OBSERVABLE_INCLUDE",stim.target_rec(-1), 0)
    return circuit
    
def measure_ZZ(circuit : stim.Circuit) -> stim.Circuit:
    circuit.append("MPP",
                   stim.target_combined_paulis([stim.target_pauli(qubit.idx, "Z")
                    for qubit in surface_codes[0] + surface_codes[1] 
                    if qubit.ancilla_type() == 'D'
                    ]))
    circuit.append("OBSERVABLE_INCLUDE",stim.target_rec(-1), 1)
    return circuit


def measure_all_stabs_non_expanded(circuit : stim.Circuit) -> stim.Circuit:
    for syn in ancilla_measurement_order:
        circuit.append("MPP",
                       stim.target_combined_paulis([stim.target_pauli(qubit.idx, syn.ancilla_type())
                                                    for qubit in syn.get_stabilizer_support(all_non_expanded_qubits)
                                                    ]))
        circuit.append("DETECTOR",stim.target_rec(-1))
    return circuit

#%%
def init_expanded_codes(circuit : stim.Circuit, 
                   max_radius, 
                   post_selected_rounds) -> stim.Circuit:
       
    circuit.rx(x_ancillas_expanded + x_init_data)
    circuit.r(z_ancillas_expanded + z_init_data)
        
    circuit.tick()
    cycles = [
            sum([qubit.get_surface_code_cx(idx, all_expanded_qubits) for qubit 
                 in sum(expanded_surface_codes, [])
                 if qubit.ancilla_type() in 'XZ'], [])
            for idx in range(4)
        ]
    for cycle in cycles:
        circuit.cx(cycle)
        circuit.tick()
    
    circuit.mrx(x_ancillas_expanded)
    circuit.mr(z_ancillas_expanded)
    circuit.tick()
    
    determined_ancillas_x = [
        q for q in x_ancillas_expanded 
        if q.support_contained(x_init_data + all_non_expanded_qubits, all_expanded_qubits)
    ]
    determined_ancillas_z = [
        q for q in z_ancillas_expanded 
        if q.support_contained(z_init_data+all_non_expanded_qubits, all_expanded_qubits)
    ]
    
    for q in determined_ancillas_x + determined_ancillas_z:
        circuit.append("DETECTOR", stim.target_rec(measurement_order_expanded.index(q) - len(measurement_order_expanded)),
                      [(q.radius < max_radius) and (index < post_selected_rounds), q.x, q.y])
        
    return circuit

def generate_ghz_cycles():
    if d == 3:
        right_qubit = find_qubit(mid_ghz_qubit, [1,0], all_non_expanded_qubits)
        left_qubit = find_qubit(mid_ghz_qubit, [-1,0], all_non_expanded_qubits)
        
        ghz_cycles = [
            [mid_ghz_qubit, right_qubit],
            [mid_ghz_qubit, left_qubit],
            sum([[q, find_qubit(q, [0, 1], all_non_expanded_qubits)] for q in [right_qubit, left_qubit, mid_ghz_qubit]],[]),
            sum([[q, find_qubit(q, [0,-1], all_non_expanded_qubits)] for q in [right_qubit, left_qubit, mid_ghz_qubit]],[]),
        ]
    elif d == 5:
        right_qubit = find_qubit(mid_ghz_qubit, [2,0], all_non_expanded_qubits)
        mid_line = [mid_ghz_qubit, find_qubit(mid_ghz_qubit, [-1,0], all_non_expanded_qubits), 
                    right_qubit, find_qubit(right_qubit,[1,0], all_non_expanded_qubits)]
        
        up_line = [find_qubit(q, [0, 1], all_non_expanded_qubits) for q in mid_line]
        down_line = [find_qubit(q, [0, -1], all_non_expanded_qubits) for q in mid_line]
        
        top_line = [find_qubit(q, [0, 2], all_non_expanded_qubits) for q in mid_line]
        bottom_line = [find_qubit(q, [0, -2], all_non_expanded_qubits) for q in mid_line]
        
        cross_line = [find_qubit(mid_ghz_qubit, [1,0 + step], all_non_expanded_qubits) for step in [-2, -1, 0, 1, 2]]
        
        ghz_cycles = [
            [mid_ghz_qubit, right_qubit],
            mid_line,
            list(sum(zip(mid_line, up_line),tuple())),
            list(sum(zip(up_line, top_line, mid_line, down_line),tuple())),
            list(sum(zip(down_line, bottom_line, mid_line, cross_line), tuple()) + (up_line[0], cross_line[-1])),
        ]
    else:
        raise NotImplemented()
    return ghz_cycles


def gidney_cycle(circuit : stim.Circuit) -> stim.Circuit:
    all_ancillas = sum(ancilla_x_qubits + ancilla_z_qubits,[])
    ghz_cycles = generate_ghz_cycles()

    inverse_ghz_cycle = ghz_cycles[::-1]
    cycles = [
            sum([qubit.get_surface_code_cx(idx, all_non_expanded_qubits) for qubit in all_ancillas], [])
            for idx in range(4)
    ]

    transversal_cx(circuit, d=d)
    circuit.tick()
    for ghz_cycle in inverse_ghz_cycle:
        circuit.cx(ghz_cycle)
        circuit.tick()
        
    circuit.mrx([mid_ghz_qubit])
    circuit.tick()
    circuit.append("DETECTOR",stim.target_rec(-1))
    
    for ghz_cycle in ghz_cycles:
        circuit.cx(ghz_cycle)
        circuit.tick()
    
    transversal_cx(circuit, d=d)
    circuit.tick()
    
    circuit.mrx([mid_ghz_qubit]+non_mid_ghz_qubit)
    circuit.tick()
    for idx in range(len(ghz_qubits)):
        circuit.append("DETECTOR",stim.target_rec(-1-idx))


    for cycle in cycles:
        circuit.cx(cycle)
        circuit.tick()
        
    circuit.mr([q for q in ancilla_measurement_order if q.ancilla_type() == 'Z'] )
    circuit.mrx([q for q in ancilla_measurement_order if q.ancilla_type() == 'X'] )
    circuit.tick()
    
    for idx in range(len(all_ancillas)):
        circuit.append("DETECTOR",stim.target_rec(-1-idx))
    
    return circuit
 
#%%
def expanded_cycle(circuit : stim.Circuit, index, 
                   max_radius, 
                   post_selected_rounds,
                   p = 0) -> stim.Circuit:
    cycles = [
            sum([qubit.get_surface_code_cx(idx, all_expanded_qubits) for qubit 
                 in sum(expanded_surface_codes, [])
                 if qubit.ancilla_type() in 'XZ'], [])
            for idx in range(4)
        ]
    
    for cycle in cycles:
        circuit.cx(cycle)
        if p > 0:
            circuit.append("DEPOLARIZE2", [i.idx for i in cycle], p)
        circuit.tick()
        
    circuit.mrx(x_ancillas_expanded)
    circuit.mr(z_ancillas_expanded)
    circuit.tick()
        
    for q in x_ancillas_expanded + z_ancillas_expanded:
        circuit.append("DETECTOR", [stim.target_rec(measurement_order_expanded.index(q) - len(measurement_order_expanded)),
                                    stim.target_rec(measurement_order_expanded.index(q) - 2*len(measurement_order_expanded)),
                                    ], [(q.radius < max_radius) and (index < post_selected_rounds), q.x, q.y])
    return circuit
       
    
#%%
def generate_unexpanded(ghz_meas_count, p = None, apply_transversal_cx = True):
    circuit = stim.Circuit()
    circuit = init_circuit(circuit, random_init = apply_transversal_cx)

    for _ in range(ghz_meas_count):
        circuit = stabilizer_ghz_cycle(circuit, p = p, apply_transversal_cx = apply_transversal_cx)
        
    circuit = measure_all_stabs_non_expanded(circuit)
    circuit = measure_ZZ(circuit)
    circuit = measure_XX(circuit)
    return circuit

def generate_expanded(ghz_meas_count, max_radius, post_selected_rounds, p = None):
    circuit = stim.Circuit()
    init_circuit(circuit)
    
    for _ in range(ghz_meas_count):
        stabilizer_ghz_cycle(circuit, p = p)
    
    init_expanded_codes(circuit, max_radius = max_radius,
                       post_selected_rounds = post_selected_rounds)
    for idx in range(d2):
        expanded_cycle(circuit, index = idx, max_radius = max_radius,
                       post_selected_rounds = post_selected_rounds)
    
    measure_all_stabs(circuit)
    measure_ZZ_expanded(circuit)
    measure_XX_expanded(circuit)
    return circuit


def generate_unexpanded_gidney(ghz_meas_count):
    circuit = stim.Circuit()
    circuit = init_circuit(circuit, gidney_style = True)

    for _ in range(ghz_meas_count):
        circuit = gidney_cycle(circuit)
        
    circuit = measure_all_stabs_non_expanded(circuit)
    circuit = measure_ZZ(circuit)
    circuit = measure_XX(circuit)
    return circuit

def generate_expanded_gidney(ghz_meas_count, max_radius, post_selected_rounds):
    circuit = stim.Circuit()
    circuit = init_circuit(circuit, gidney_style = True)

    for _ in range(ghz_meas_count):
        circuit = gidney_cycle(circuit)
        
    stabilizer_ghz_cycle(circuit) 
    init_expanded_codes(circuit)
    for idx in range(d2):
        expanded_cycle(circuit, index = idx, max_radius = max_radius,
                       post_selected_rounds = post_selected_rounds)
    
    measure_all_stabs(circuit)
    circuit = measure_ZZ(circuit)
    circuit = measure_XX(circuit)
    return circuit



#%%
parallel = Parallel(n_jobs=-1)
if __name__ == '__main__':

    def save_circ(p, num_ghz_measurements):
        stim.Circuit.cx = lambda self, targets: self.append("CX", [i.idx for i in targets])
        stim.Circuit.swap = lambda self, targets: self.append("SWAP", [i.idx for i in targets])
        stim.Circuit.h = lambda self, targets: self.append("H", [i.idx for i in targets])
        stim.Circuit.r = lambda self, targets: self.append("R", [i.idx for i in targets])
        stim.Circuit.rx = lambda self, targets: self.append("RX", [i.idx for i in targets])
        stim.Circuit.mr = lambda self, targets: self.append("MR", [i.idx for i in targets])
        stim.Circuit.mrx = lambda self, targets: self.append("MRX", [i.idx for i in targets])
        stim.Circuit.tick = lambda self: self.append("TICK")
        
        for noise, noise_name in zip([gen.NoiseModel.uniform_depolarizing(p), gen.NoiseModel.uniform_depolarizing_neutral_atoms(p)], ["uniform","uniform_atoms"]):
            circuit = generate_unexpanded(num_ghz_measurements, p = p)
            noisey_circuit = noise.noisy_circuit_skipping_mpp_boundaries(circuit)
            noisey_circuit.to_file(
                f"circuits/c=init,d1={d},p={p},num_ghz_measurements={num_ghz_measurements},noise={noise_name}.stim")


    ps = np.array([2e-3, 1e-3, 5e-4, 3e-4, 2e-4, 1e-4])
    parallel(delayed(save_circ)(p, num_ghz_measurements)
             for p in tqdm(ps)
             for num_ghz_measurements in tqdm([d-1,d], leave=False))

    def save_circ(p, num_ghz_measurements):
        stim.Circuit.cx = lambda self, targets: self.append("CX", [i.idx for i in targets])
        stim.Circuit.swap = lambda self, targets: self.append("SWAP", [i.idx for i in targets])
        stim.Circuit.h = lambda self, targets: self.append("H", [i.idx for i in targets])
        stim.Circuit.r = lambda self, targets: self.append("R", [i.idx for i in targets])
        stim.Circuit.rx = lambda self, targets: self.append("RX", [i.idx for i in targets])
        stim.Circuit.mr = lambda self, targets: self.append("MR", [i.idx for i in targets])
        stim.Circuit.mrx = lambda self, targets: self.append("MRX", [i.idx for i in targets])
        stim.Circuit.tick = lambda self: self.append("TICK")

        
        for noise, noise_name in zip([gen.NoiseModel.uniform_depolarizing(p), gen.NoiseModel.uniform_depolarizing_neutral_atoms(p)], ["uniform","uniform_atoms"]):

            circuit = generate_expanded(num_ghz_measurements, 0, 0, p = p)
            noisey_circuit = noise.noisy_circuit_skipping_mpp_boundaries(circuit)
            noisey_circuit.to_file(
                f"circuits/c=end2end,d1={d},d2={d2},p={p},num_ghz_measurements={num_ghz_measurements},noise={noise_name}.stim")


    ps = np.array([2e-3, 1e-3, 5e-4, 3e-4, 2e-4, 1e-4])
    parallel(delayed(save_circ)(p, num_ghz_measurements)
             for p in tqdm(ps) for num_ghz_measurements  in [d])


    
