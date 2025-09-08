from __future__ import annotations
from typing import Iterator
import numpy as np
import qutip as qp
try:
    from qutip.qip.operations.gates import cnot, toffoli, controlled_gate,hadamard_transform,y_gate, x_gate, z_gate, expand_operator
except:
    pass
import matplotlib.pyplot as plt
from tqdm import tqdm
import stim
from itertools import product
import pennylane as qml
import random
from functools import reduce
try:
    import cirq
    import qsimcirq
except:
    pass

try:
    import qiskit as qk
    from qiskit.circuit.library import HGate, CCXGate, CXGate, CHGate, XGate, YGate, ZGate
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from qiskit.quantum_info.operators import Pauli
except:
    pass

import sys
sys.path.append("../StateVecSimulator/")
sys.path.append("../StateVecSimulator/latte/")
try: 
    from vec_sim import convertion_dict
except:
    print("Vec_sim not found, some features will be disabled")

up, down = qp.basis(2,0), qp.basis(2,1)
down_x, up_x = qp.sigmax().eigenstates()[1]
down_y, up_y = qp.sigmay().eigenstates()[1]
identity = qp.identity(2)

states_dict = {"z" : up, "x" : up_x, "y" : up_y}
pauli_dict = {"x" : qp.sigmax(), "y" : qp.sigmay(), "z" : qp.sigmaz()}
identity = qp.identity(2)

rotated_order_x = [(-1,-1), (1,-1), (-1,1), (1,1)]
rotated_order_z = [(-1,-1), (-1,1), (1,-1), (1,1)]

unrotated_order_x  = unrotated_order_z = [(1,0), (0,1), (0, -1), (-1,0)]


class QubitList(list):
    def plot(self, *args, **kwargs):
        if len(self) == 0:
            return
        plt.scatter(*zip(*self), *args, **kwargs)

class Qubit:
    def __init__(self, x, y, simulation : Simulation):
        self.x : int = x
        self.y : int = y
        self.simulation : Simulation = simulation

    def __sub__(self, other : Qubit):
        return self.x-other.x, self.y-other.y

    def __iter__(self):
        return iter((self.x, self.y))

    def __repr__(self):
        return f"Qubit(({self.x}, {self.y}))"

    @property
    def index(self):
        return self.simulation.get_qubit_index(self)

    def is_data(self):
        return (self.x + self.y) % 2 == 0

    @property
    def basis(self):
        if (self.x % 2 == 1 and self.y % 2 == 0):
            return "x"
        elif (self.x % 2 == 0 and self.y % 2 == 1):
            return "z"
        else:
            raise Exception(f"Basis not defined for Qubit {self}")

    def basis_operator(self):
        return pauli_dict[self.basis]

    def get_support(self) -> QubitList:
        assert not self.is_data()
        support = []
        for qubit in self.simulation.data_qubits:
            if sum(np.abs(qubit - self)) == 1:
                support.append(qubit)
        return QubitList(support)
    
    def single_support_qubit(self):
        support = self.get_support()
        ancillas = self.simulation.x_ancilla if self in self.simulation.x_ancilla else self.simulation.z_ancilla
        for qubit in support:
            for ancilla in ancillas:
                if qubit in ancilla.get_support() and ancilla != self:
                    break 
            else:
                return qubit
        raise Exception(f"No single support qubit for {self}")
    
    def get_correction(self):
        correction_length = (self.simulation.d - 1)//2 if self.simulation.d%2 == 1 else self.simulation.d//2
        if correction_length == 1:
            return [self.single_support_qubit()]
        elif correction_length == 2:
            try :
                return [self.single_support_qubit()]
            except:
                for qubit in self.get_support():
                    for ancilla in qubit.get_all_stabilizers(basis = self.basis):
                        try:
                            return [qubit, ancilla.single_support_qubit()]
                        except:
                            pass
                else:
                    raise Exception(f"No correction found for {self}")
        else:
            raise Exception(f"Correction length {correction_length} not implemented")

    def get_stabilizer_operator(self):
        if self.is_data():
            raise Exception("No stabilizer for data qubits")

        return self.simulation.operator({
            qubit.index : self.basis_operator()
            for qubit in self.get_support()
        })

    def get_projector(self, sign = 1):
        return (self.simulation.identity + sign*self.get_stabilizer_operator())/2

    def get_single_qubit_projector(self, basis, sign = 1):
        return (self.simulation.identity + sign*self.simulation.operator({self.index : basis}))/2
    
    def get_all_stabilizers(self , basis : str):
        if basis.lower() == "x":
            return QubitList([ancilla for ancilla in self.simulation.x_ancilla if self in ancilla.get_support()])
        elif basis.lower() == "z":
            return QubitList([ancilla for ancilla in self.simulation.z_ancilla if self in ancilla.get_support()])
        else:
            raise Exception(f"Unknown stabilizer basis {basis}")

   

class Simulation:
    def __init__(self, d):
        self.d = d
        self.ancilla_offset = 2*d + 2

        self.data_qubits : QubitList[Qubit] = QubitList([
            Qubit(x, y, self)
            for x in range (2*d - 1)
                for y in range(2*d - 1)
            if (x + y) % 2 == 0
        ])

        self.ghz_ancilla : QubitList[Qubit] = QubitList([
            Qubit(x + self.ancilla_offset, y, self)
            for x in range (2*d - 1)
                for y in range(2*d - 1)
            if (x + y) % 2 == 0 and x <= y and ((x != y) or x % 2 == 0)
        ])

        self.x_ancilla : QubitList[Qubit] = QubitList([
            Qubit(x, y, self)
            for x in range (2*d - 1)
                for y in range(2*d - 1)
            if (x%2 == 1 and y%2 == 0)
        ])
        self.z_ancilla : QubitList[Qubit] = QubitList([
            Qubit(x, y, self)
            for x in range (2*d - 1)
                for y in range(2*d - 1)
            if (x%2 == 0 and y%2 == 1)
        ])

        self.residual_qubits = QubitList([
            Qubit(self.ancilla_offset + 2*d + idx, 0, self)
            for idx in range(len(self.x_ancilla + self.z_ancilla) - len(self.ghz_ancilla))
        ])

        self.extra_qubits = QubitList([]) # Used only for expansion
        self.generate_qubit_lists()
        self.generate_qubit_dict()

        self.identity = 1
        self.logical_z_qubits = QubitList([qubit for qubit in self.data_qubits if qubit.y == 0 ])
        self.logical_x_qubits = QubitList([qubit for qubit in self.data_qubits if qubit.x == 0 ])
        try:
            self.cirq_qubits = [cirq.LineQubit(i) for i in range(self.count_simulated_qubits())]
        except :
            pass
        
    def generate_qubit_lists(self):
        self.qubits_to_simulate = self.data_qubits + self.ghz_ancilla + self.residual_qubits + self.extra_qubits
        self.all_qubits = self.data_qubits + self.ghz_ancilla + self.x_ancilla + self.z_ancilla + self.residual_qubits + self.extra_qubits

    def generate_qubit_dict(self):
        self.qubit_dict = {i:qubit for i,qubit in enumerate(self.qubits_to_simulate)}

    def get_qubit(self, x : int, y : int, return_none_if_missing = False) -> Qubit:
        for qubit in self.all_qubits:
            if qubit.x == x and qubit.y == y:
                return qubit
        if return_none_if_missing:
            return None
        raise ValueError(f"No qubit at that position ({x},{y})")
    
    def is_corner_qubit(self, qubit : Qubit):
        if qubit not in self.data_qubits:
            raise Exception(f"Qubit {qubit} is not a data qubit, can't check if it's a corner qubit")
        
        elif self.d == 2:
            if ((qubit.x == 0 or qubit.y == 2*self.d - 2) and not (qubit.x == 0 and qubit.y == 2*self.d - 2)):
                return True
            return False
        
        elif self.d == 3:
            if (qubit.x == 0 or qubit.x == 2*self.d - 2) and (qubit.y == 0 or qubit.y == 2*self.d - 2):
                return True
            return False
        elif self.d == 5:
            if ((qubit.x == 0 or qubit.x == 2*self.d - 2) or (qubit.y == 0 or qubit.y == 2*self.d - 2)) and 4 not in [qubit.x, qubit.y]:
                return True
            if qubit.x in [1, 7] and qubit.y in [1, 7]:
                return True
            return False
        else:
            raise Exception(f"Don't know how to check if {qubit} is a corner qubit for d = {self.d}")

    def count_simulated_qubits(self):
        return len(self.qubits_to_simulate)
    
    def get_qubit_index(self, qubit : Qubit) -> int:
        # First try to find in qubit_dict
        for idx, q in self.qubit_dict.items():
            if q == qubit:
                return idx
        
        # If not found, check if it's a measurement ancilla (x or z)
        measurement_ancillas = self.x_ancilla + self.z_ancilla
        for idx, q in enumerate(measurement_ancillas):
            if q == qubit:
                # Map to corresponding index in ghz+residual qubits
                mapped_qubits = self.ghz_ancilla + self.residual_qubits
                if idx < len(mapped_qubits):
                    return self.get_qubit_index(mapped_qubits[idx])
                break
        
        raise Exception(f"Can't find qubit index {qubit}")

    def preplot(self):
        plt.xlim(-1, self.ancilla_offset + 2 * self.d)
        plt.ylim(-1, 2 * self.d)
        plt.grid()

    def plot(self):
        self.data_qubits.plot(label = "Data")
        self.x_ancilla.plot(label = "X Stabilizers")
        self.z_ancilla.plot(label = "Z Stabilizers")
        self.ghz_ancilla.plot(label = "GHZ Ancilla")
        self.residual_qubits.plot(label = "Residual")


    def get_state(self, basis_string = None):
        if basis_string is None:
            basis_string = "z"* self.count_simulated_qubits()
        basis_string = basis_string.lower()
        if len(basis_string) != self.count_simulated_qubits():
            raise ValueError(f"Basis string {len(basis_string)} does not match number of qubits {self.count_simulated_qubits()}")

        return qp.tensor([
            states_dict[basis]
            for basis in basis_string
        ])

    def operator(self, operator_dictionary: dict[int, qp.Qobj]) -> qp.Qobj:
        return qp.tensor(
            [
                operator_dictionary.get(qubit, identity)
                for qubit in range(self.count_simulated_qubits())
            ]
        )

    def cnot(self, control, target):
        return cnot(self.count_simulated_qubits(),
        self.get_qubit_index(control),
        self.get_qubit_index(target))

    def x_gate(self, qubit : Qubit):
        return x_gate(self.count_simulated_qubits(), self.get_qubit_index(qubit))

    def y_gate(self, qubit : Qubit):
        return y_gate(self.count_simulated_qubits(), self.get_qubit_index(qubit))

    def z_gate(self, qubit : Qubit):
        return z_gate(self.count_simulated_qubits(), self.get_qubit_index(qubit))

    def h(self, qubit : Qubit):
        return expand_operator(hadamard_transform(1), self.count_simulated_qubits(), self.get_qubit_index(qubit))

    def toffoli(self, control_a : Qubit, control_b : Qubit, target : Qubit):
        return toffoli(self.count_simulated_qubits(),
                              [self.get_qubit_index(control_a), self.get_qubit_index(control_b)],
                              self.get_qubit_index(target))

    def control_swap(self, control : Qubit, target_a : Qubit, target_b : Qubit):
        return self.toffoli(control, target_a, target_b) * self.toffoli(control, target_b, target_a) * self.toffoli(control, target_a, target_b)

    def ch(self, control : Qubit, target : Qubit):
        return controlled_gate(hadamard_transform(1),
                               self.get_qubit_index(control),
                               self.get_qubit_index(target),
                               self.count_simulated_qubits())


    def multi_qubit_pauli(self, qubits : list[Qubit], pauli : str):
        op = self.identity
        for qubit in qubits:
            if pauli.lower() == 'x':
                op *= self.x_gate(qubit)
            elif pauli.lower() == 'y':
                op *= self.y_gate(qubit)
            elif pauli.lower() == 'z':
                op *= self.z_gate(qubit)
            else:
                raise Exception(f"Unknown pauli {pauli}")
        return op

    def logical_x(self):
        return self.multi_qubit_pauli(self.logical_x_qubits, "x")
    def logical_z(self):
        return self.multi_qubit_pauli(self.logical_z_qubits, "z")
    def get_both_logical_operators(self) -> QubitList:
        # Return qubits in order: common qubit, z qubits, x qubits
        common_qubit = None
        for x_qubit in self.logical_x_qubits:
            if x_qubit in self.logical_z_qubits:
                common_qubit = x_qubit
                break
        
        # Get remaining qubits excluding common qubit
        z_qubits = [q for q in self.logical_z_qubits if q != common_qubit]
        x_qubits = [q for q in self.logical_x_qubits if q != common_qubit]
        
        return QubitList([common_qubit] + z_qubits + x_qubits)
    
    def prepare_x_rotation(self, circuit : Circuit, unitary = None):
        if self.d == 3: 
            xz, z1, z2, x1, x2 = self.get_both_logical_operators()
            circuit.append(CX(self, x1, x2, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, x1, xz, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(H(self, x1, unitary = unitary))
            circuit.append(Tick(self))
        elif self.d == 5:
            xz, z1, z2, z3, z4, x1, x2, x3, x4 = self.get_both_logical_operators()
            circuit.append(CX(self, x1, xz, unitary = unitary))
            circuit.append(CX(self, x3, x4, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, x2, x1, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, x2, x3, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(H(self, x2, unitary = unitary))
            circuit.append(Tick(self))
        else:
            raise Exception(f"Don't know how to prepare x rotation for d = {self.d}")

    def unprepare_x_rotation(self, circuit : Circuit, unitary = None, final_tick = True):
        if self.d == 3:
            xz, z1, z2, x1, x2 = self.get_both_logical_operators()
            circuit.append(H(self, x1, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, x1, xz, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, x1, x2, unitary = unitary))
        elif self.d == 5:
            xz, z1, z2, z3, z4, x1, x2, x3, x4 = self.get_both_logical_operators()
            circuit.append(H(self, x2, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, x2, x3, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, x3, x4, unitary = unitary))
            circuit.append(CX(self, x2, x1, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, x1, xz, unitary = unitary))
 
        else:
            raise Exception(f"Don't know how to unprepare x rotation for d = {self.d}")
        if final_tick:
            circuit.append(Tick(self))

    def prepare_z_rotation(self, circuit : Circuit, unitary = None):
        if self.d == 3:
            xz, z1, z2, x1, x2 = self.get_both_logical_operators()
            circuit.append(CX(self, xz, z1, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, z2, z1, unitary = unitary))
            circuit.append(Tick(self))
        elif self.d == 5:
            xz, z1, z2, z3, z4, x1, x2, x3, x4 = self.get_both_logical_operators()
            circuit.append(CX(self, z4, z3, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, z3, z2, unitary = unitary))
            circuit.append(CX(self, xz, z1, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, z1, z2, unitary = unitary))
            circuit.append(Tick(self))           
        else:
            raise Exception(f"Don't know how to prepare z rotation for d = {self.d}")

    def unprepare_z_rotation(self, circuit : Circuit, unitary = None, final_tick = True):
        if self.d == 3:
            xz, z1, z2, x1, x2 = self.get_both_logical_operators()
            circuit.append(CX(self, z2, z1, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, xz, z1, unitary = unitary))
        elif self.d == 5:
            xz, z1, z2, z3, z4, x1, x2, x3, x4 = self.get_both_logical_operators()
            circuit.append(CX(self, z1, z2, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, z3, z2, unitary = unitary))
            circuit.append(CX(self, xz, z1, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, z4, z3, unitary = unitary))
        else:
            raise Exception(f"Don't know how to unprepare z rotation for d = {self.d}")
            
        if final_tick:
            circuit.append(Tick(self))
        

    def generate_injection_circuit(self, generate_unitary = False):
        unitary = None if generate_unitary else self.identity
        circuit = Circuit(self)
        if self.d == 2:
            xz, z1, x1 = self.get_both_logical_operators()
            circuit.append(CX(self, x1, xz, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(H(self, x1, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(T(self, xz, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(H(self, x1, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, x1, xz, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, xz, z1, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(S(self, z1, unitary = unitary))
            circuit.append(Tick(self))
            circuit.append(CX(self, xz, z1, unitary = unitary))
            circuit.append(Tick(self))
            return circuit

        if self.d in [3, 5]:
            if self.d == 3:
                xz, z1, z2, x1, x2 = self.get_both_logical_operators()
                rotation_qubit_x = x1
                rotation_qubit_z = z1
            elif self.d == 5:
                xz, z1, z2, z3, z4, x1, x2, x3, x4 = self.get_both_logical_operators()
                rotation_qubit_x = x2
                rotation_qubit_z = z2

            self.prepare_x_rotation(circuit, unitary)
            circuit.append(T(self, rotation_qubit_x, unitary = unitary))
            circuit.append(Tick(self))
            self.unprepare_x_rotation(circuit, unitary, final_tick = False)

            self.prepare_z_rotation(circuit, unitary)
            circuit.append(S(self, rotation_qubit_z, unitary = unitary))
            circuit.append(Tick(self))
            self.unprepare_z_rotation(circuit, unitary, final_tick = True)

        else:
            raise Exception(f"Don't know how to inject ancilla for d = {self.d}")
        return circuit
    
    def generate_injection_circuit_HXY(self, init_basis : str, generate_unitary = False):
        unitary = None if generate_unitary else self.identity
        circuit = Circuit(self)
        init_basis = init_basis.lower()
        if self.d in [3, 5]:
            if self.d == 3:
                xz, z1, z2, x1, x2 = self.get_both_logical_operators()
                rotation_qubit_x = x1
                rotation_qubit_z = z1
            elif self.d == 5:
                xz, z1, z2, z3, z4, x1, x2, x3, x4 = self.get_both_logical_operators()
                rotation_qubit_x = x2
                rotation_qubit_z = z2

            if init_basis == 'z':
                self.prepare_x_rotation(circuit, unitary)
                circuit.append(Sdag(self, rotation_qubit_x, unitary = unitary))
                circuit.append(Tick(self))
                self.unprepare_x_rotation(circuit, unitary, final_tick = False)

                self.prepare_z_rotation(circuit, unitary)
                circuit.append(Tdag(self, rotation_qubit_z, unitary = unitary))
                circuit.append(Tick(self))
                self.unprepare_z_rotation(circuit, unitary, final_tick = True)
            elif init_basis == 'x':
                self.prepare_z_rotation(circuit, unitary)
                circuit.append(T(self, rotation_qubit_z, unitary = unitary))
                circuit.append(Tick(self))
                self.unprepare_z_rotation(circuit, unitary, final_tick = True)
            else:
                raise Exception(f"Invalid basis: {init_basis}")
 
        else:
            raise Exception(f"Don't know how to inject ancilla H_XY for d = {self.d}")
        return circuit
    
    def expand_ancilla_step(self, circuit : Circuit, step : int, unitary = None):
        if self.d == 2:
            expanded_qubits = [self.ghz_ancilla[0]]
        else:
            expanded_qubits = [self.ghz_ancilla[3]]
        step_idx = 0 
        while len(expanded_qubits) < len(self.ghz_ancilla):
            current_round_CX_pairs = []
            current_round_expanded = list(expanded_qubits)
            for qubit in current_round_expanded:
                for target in self.ghz_ancilla:
                    if target not in expanded_qubits:
                        current_round_CX_pairs.append((qubit, target))
                        expanded_qubits.append(target)
                        break
            if step_idx == step:
                for cx_pair in current_round_CX_pairs:
                    circuit.append(CX(self, cx_pair[0], cx_pair[1], unitary = unitary))
                break
            step_idx += 1
        else:
            raise Exception(f"Expansion for {self.d=} is done after step {step_idx}, got {step=}")
           

    def expansion_length(self):
        return int(np.ceil(np.log2(len(self.ghz_ancilla)+1)))
    
    def expand_ancilla_for_circuit(self, circuit : Circuit, unitary = None):
        for idx in range(self.expansion_length()):
            self.expand_ancilla_step(circuit, idx, unitary)
            circuit.append(Tick(self))
    def collapse_ancilla_for_circuit(self, circuit : Circuit, unitary = None):
        for idx in range(self.expansion_length()):
            self.expand_ancilla_step(circuit, self.expansion_length() - idx - 1, unitary)
            circuit.append(Tick(self))
    
    def phase_kickback(self, circuit : Circuit, unitary = None):
        circuit.append(BeginPhaseKickback(self))
        for ancilla in self.ghz_ancilla:
            data_q = self.get_qubit(ancilla.x - self.ancilla_offset, ancilla.y)
            circuit.append(CH(self, ancilla, data_q, unitary = unitary))

        circuit.append(Tick(self))

        for ancilla in self.ghz_ancilla:
            if not (ancilla.x - self.ancilla_offset == ancilla.y):
                # Off diagonal
                data_q = self.get_qubit(ancilla.y, ancilla.x - self.ancilla_offset)
                circuit.append(CH(self, ancilla, data_q, unitary = unitary))
            else :
                # On Diagonal
                if ancilla.y != 0:
                    data_q = self.get_qubit(ancilla.y - 1, ancilla.x - self.ancilla_offset - 1)
                    circuit.append(CH(self, ancilla, data_q, unitary = unitary))

        circuit.append(Tick(self))

        for ancilla in self.ghz_ancilla:
            if ancilla.x - self.ancilla_offset == ancilla.y:
                pass
            else:
                data_q1 = self.get_qubit(ancilla.x - self.ancilla_offset, ancilla.y)
                data_q2 = self.get_qubit(ancilla.y, ancilla.x - self.ancilla_offset)
                circuit.append(CSwap(self, ancilla, data_q1, data_q2, unitary = unitary))

        circuit.append(Tick(self))

    def phase_kickback_HXY(self, circuit : Circuit, unitary = None, inverse = False):
        circuit.append(BeginPhaseKickback(self))
        for ancilla in self.ghz_ancilla:
            if ancilla.x - self.ancilla_offset == ancilla.y:
                if ancilla.y % 2 == 0:
                    data_q = self.get_qubit(ancilla.x - self.ancilla_offset, ancilla.y )
                    circuit.append(CS_dagX(self, ancilla, data_q, unitary = unitary))
            else:
                data_q1 = self.get_qubit(ancilla.x - self.ancilla_offset, ancilla.y)
                data_q2 = self.get_qubit(ancilla.y, ancilla.x - self.ancilla_offset)
                circuit.append(CCZ(self, ancilla, data_q1, data_q2, unitary = unitary))
        circuit.append(Tick(self))

        for ancilla in self.ghz_ancilla:
            if ancilla.x - self.ancilla_offset == ancilla.y:
                if ancilla.y % 2 == 0 and ancilla.y < self.d*2 - 2:
                    data_q = self.get_qubit(ancilla.x - self.ancilla_offset+1, ancilla.y+1)
                    circuit.append(CSX(self, ancilla, data_q, unitary = unitary))
        
        circuit.append(Tick(self))

    def phase_kickback_CZ(self, circuit : Circuit, unitary = None, target_gate : Gate = None, label = True):
        if target_gate is None:
            target_gate = CZ
        if label:
            circuit.append(BeginPhaseKickback(self))
        for ancilla in self.ghz_ancilla:
            if ancilla.x - self.ancilla_offset == ancilla.y:
                if ancilla.y % 2 == 0:
                    data_q = self.get_qubit(ancilla.x - self.ancilla_offset, ancilla.y )
                else :
                    data_q = None
            else:
                data_q = self.get_qubit(ancilla.x - self.ancilla_offset, ancilla.y)
            if data_q is not None and (not self.is_corner_qubit(data_q)):
                circuit.append(target_gate(self, ancilla, data_q, unitary = unitary))
        circuit.append(Tick(self))

        for ancilla in self.ghz_ancilla:
            if ancilla.x - self.ancilla_offset == ancilla.y:
                if ancilla.y % 2 == 0 and ancilla.y < self.d*2 - 2:
                    data_q = self.get_qubit(ancilla.x - self.ancilla_offset + 1, ancilla.y + 1 )
                else :
                    data_q = None
            else:
                data_q = self.get_qubit(ancilla.y, ancilla.x - self.ancilla_offset)
            if data_q is not None and (not self.is_corner_qubit(data_q)):
                circuit.append(target_gate(self, ancilla, data_q, unitary = unitary))
        circuit.append(Tick(self))


    def generate_projection_circuit(self, generate_unitary = False):
        circuit = Circuit(self)
        unitary = None if generate_unitary else self.identity
        central_ancilla_id = 3 if self.d != 2 else 0

        circuit.append(H(self,self.ghz_ancilla[central_ancilla_id], unitary = unitary))
        circuit.append(Tick(self))

        self.expand_ancilla_for_circuit(circuit, unitary)

        self.phase_kickback(circuit, unitary)

        self.collapse_ancilla_for_circuit(circuit, unitary)

        circuit.append(H(self,self.ghz_ancilla[central_ancilla_id], unitary = unitary))
        circuit.append(Tick(self))

        for qubit in self.ghz_ancilla:
            circuit.append(MeasureReset(self, qubit, projector = unitary))
        for qubit in self.residual_qubits:
            circuit.append(Reset(self, qubit))
        circuit.append(Tick(self))

        return circuit
    
    def generate_projection_circuit_HXY(self, generate_unitary = False):
        circuit = Circuit(self)
        unitary = None if generate_unitary else self.identity

        circuit.append(H(self, self.ghz_ancilla[3], unitary = unitary))
        circuit.append(Tick(self))

        self.expand_ancilla_for_circuit(circuit, unitary)
        self.phase_kickback_HXY(circuit, unitary)

        self.collapse_ancilla_for_circuit(circuit, unitary)

        circuit.append(H(self, self.ghz_ancilla[3], unitary = unitary))
        circuit.append(Tick(self))

        for qubit in self.ghz_ancilla:
            circuit.append(MeasureReset(self, qubit, projector = unitary))
        for qubit in self.residual_qubits:
            circuit.append(Reset(self, qubit))
        circuit.append(Tick(self))
        return circuit

    def generate_double_ghz(self, generate_unitary = False):
        circuit = Circuit(self)
        unitary = None if generate_unitary else self.identity
        central_ancilla_id = 3 if self.d != 2 else 0

        for qubit in self.ghz_ancilla:
            circuit.append(H(self, qubit, unitary = unitary))
        circuit.append(Tick(self))  

        self.phase_kickback(circuit, unitary)
        self.collapse_ancilla_for_circuit(circuit, unitary)

        circuit.append(MeasureResetX(self, self.ghz_ancilla[central_ancilla_id], projector = unitary))
        circuit.append(Tick(self))

        self.expand_ancilla_for_circuit(circuit, unitary)
        self.phase_kickback(circuit, unitary)
        
        for qubit in self.ghz_ancilla:
            circuit.append(H(self, qubit, unitary = unitary))
        circuit.append(Tick(self))

        for qubit in self.ghz_ancilla:
            circuit.append(MeasureReset(self, qubit, projector = unitary))
        for qubit in self.residual_qubits:
            circuit.append(Reset(self, qubit))
        circuit.append(Tick(self))
        return circuit

    def generate_double_ghz_HXY(self, generate_unitary = False):
        circuit = Circuit(self)
        unitary = None if generate_unitary else self.identity

        for qubit in self.ghz_ancilla:
            circuit.append(H(self, qubit, unitary = unitary))
        circuit.append(Tick(self))  

        self.phase_kickback_HXY(circuit, unitary)
        self.collapse_ancilla_for_circuit(circuit, unitary)

        circuit.append(MeasureResetX(self, self.ghz_ancilla[3], projector = unitary))
        circuit.append(Tick(self))

        self.expand_ancilla_for_circuit(circuit, unitary)
        self.phase_kickback_HXY(circuit, unitary)

        for qubit in self.ghz_ancilla:
            circuit.append(H(self, qubit, unitary = unitary))
        circuit.append(Tick(self))

        for qubit in self.ghz_ancilla:
            circuit.append(MeasureReset(self, qubit, projector = unitary))
        for qubit in self.residual_qubits:
            circuit.append(Reset(self, qubit))
        circuit.append(Tick(self))
        return circuit
    
    def generate_double_ghz_CZ(self, target_gate : Gate = None, generate_unitary = False):
        circuit = Circuit(self)
        unitary = None if generate_unitary else self.identity
        for qubit in self.ghz_ancilla:
            circuit.append(ResetX(self, qubit))

        circuit.append(Tick(self))
        self.phase_kickback_CZ(circuit, unitary, label = False, target_gate = target_gate)
        self.collapse_ancilla_for_circuit(circuit, unitary)

        circuit.append(MeasureResetX(self, self.ghz_ancilla[3], projector = unitary))
        circuit.append(Tick(self))

        self.expand_ancilla_for_circuit(circuit, unitary)
        self.phase_kickback_CZ(circuit, unitary, label = False, target_gate = target_gate)

        for qubit in self.ghz_ancilla:
            circuit.append(H(self, qubit, unitary = unitary))
        circuit.append(Tick(self))

        for qubit in self.ghz_ancilla:
            circuit.append(MeasureReset(self, qubit))
        for qubit in self.residual_qubits:
            circuit.append(Reset(self, qubit))
        circuit.append(Tick(self))
        return circuit

    def generate_syndrome_circuit(self, generate_unitary = False, detect_x = True, detect_z = True):
        stim_circuit = stim.Circuit.generated(code_task="surface_code:unrotated_memory_z",
                       distance = self.d, rounds = 1)

        circuit = Circuit(self)
        unitary = None if generate_unitary else self.identity
        qubits_dictionary = {}

        for line in tqdm(list(stim_circuit), leave = False):
            if line.name == 'QUBIT_COORDS':
                qubits_dictionary[line.targets_copy()[0]] = tuple(map(int, line.gate_args_copy()))

            elif line.name == 'H':
                for qubit in line.targets_copy():
                    circuit.append(H(self, self.get_qubit(*qubits_dictionary[qubit]), unitary = unitary))
            
            elif line.name == 'CX':
                for control_qubit, target_qubit in list(zip(*[iter(line.targets_copy())]*2)):
                    circuit.append(CX(self, self.get_qubit(*qubits_dictionary[control_qubit]), self.get_qubit(*qubits_dictionary[target_qubit]), unitary = unitary))
            elif line.name == 'MR':
                for qubit in line.targets_copy():
                    ancilla_type = self.get_qubit(*qubits_dictionary[qubit]).basis
                    if ancilla_type == "x" and detect_x:
                        detect = True
                    elif ancilla_type == "z" and detect_z:
                        detect = True
                    else:
                        detect = False
                    circuit.append(MeasureReset(self, self.get_qubit(*qubits_dictionary[qubit]), 
                                                projector = unitary, detect = detect))
            elif line.name == 'TICK':
                circuit.append(Tick(self))
            else:
                pass
        circuit.append(Tick(self))
        return circuit
    
    def generate_init_circuit(self, basis : str, generate_unitary = False):
        basis = basis.lower()
        circuit = Circuit(self)
        unitary = None if generate_unitary else self.identity
        for qubit in self.data_qubits:
            circuit.append(Reset(self, qubit))
        circuit.append(Tick(self))
        for qubit in self.data_qubits:
            if basis == "x":
                circuit.append(H(self, qubit, unitary = unitary))
            elif basis == "z":
                pass
            else:
                raise ValueError(f"Invalid basis: {basis}")
        circuit.append(Tick(self))
        
        syndrome_circuit = self.generate_syndrome_circuit(detect_x = (basis == 'x'), 
                                                          detect_z = (basis == 'z'))
        return circuit + syndrome_circuit
    
    def apply_circuit(self, state: qp.Qobj, circuit: Circuit, noise: float = 0, verbose = False, loading_bar = False):
        state = state.copy()
        tqdm_ = tqdm if loading_bar else (lambda x,**kwargs:x)
        for gate in tqdm_(circuit.gates, leave = False, smoothing = 1):
            if verbose:
                print(f"Applying Gate {gate}")
            if noise != 0:
                state = gate.apply_noise(state, noise)
            else:
                state = gate.apply(state)
        return state

    def apply_statevector_trajectory(self, statevector: qp.Qobj, circuit: Circuit, noise: float = 0, verbose = False, loading_bar = False):
        statevector = statevector.copy()
        tqdm_ = tqdm if loading_bar else (lambda x,**kwargs:x)
        for gate in tqdm_(circuit.gates, leave = False, smoothing = 1):
            if verbose:
                print(f"Applying Gate {gate}")
            if noise != 0:
                statevector = gate.apply_statevector_noise(statevector, noise)
            else:
                statevector = gate.apply_statevector(statevector)
        return statevector
    
    def stim_expansion_circuit(self, d2: int, rounds : int, logical_measurement_basis : str = "z"):
        circuit = stim.Circuit()
        circuit.append("TICK")
        logical_measurement_basis = logical_measurement_basis.lower()

        x_ancilla_locations = []
        z_ancilla_locations = []
        data_qubits = []

        qubit_location_index_dict = {}
        d1_code_data_qubits = [] 

        for qubit in self.data_qubits + self.ghz_ancilla + self.residual_qubits:
            qubit_location_index_dict[(qubit.x, qubit.y)] = qubit.index
            if qubit in self.data_qubits:
                d1_code_data_qubits.append((qubit.x, qubit.y))
        
        for x in range(0, 2*d2-1):
            for y in range(0, 2*d2-1):
                if (x + y) % 2 == 1:
                    if x % 2 == 1:
                        x_ancilla_locations.append((x, y))
                    else:
                        z_ancilla_locations.append((x, y))
                else:
                    data_qubits.append((x, y))

        current_index = max(qubit_location_index_dict.values()) + 1

        x_initlized_data = []
        z_initlized_data = []
        measurement_order = x_ancilla_locations + z_ancilla_locations 

        for qubit in x_ancilla_locations + z_ancilla_locations + data_qubits :
            if qubit not in qubit_location_index_dict:
                qubit_location_index_dict[qubit] = current_index
                circuit.append("QUBIT_COORDS", current_index, qubit)
                current_index += 1

        for qubit in x_ancilla_locations:
            circuit.append("RX", qubit_location_index_dict[qubit])

        for qubit in z_ancilla_locations:
            circuit.append("R", qubit_location_index_dict[qubit])
            
        for qubit in data_qubits:
            x, y = qubit
            if qubit not in d1_code_data_qubits:
                if x >= y:
                    circuit.append("R", qubit_location_index_dict[qubit])
                    z_initlized_data.append(qubit)
                elif x < y:
                    circuit.append("RX", qubit_location_index_dict[qubit])
                    x_initlized_data.append(qubit)
        
        circuit.append("TICK")

        syndrome_cycle(circuit, x_ancilla_locations, z_ancilla_locations, data_qubits, qubit_location_index_dict, measurement_order, rotated_code=False)

        for ancilla in x_ancilla_locations + z_ancilla_locations:
            ancilla_support = ancilla_support_in_expanded_code(ancilla, data_qubits, rotated_code=False)
            original_ancillas = self.x_ancilla if ancilla in x_ancilla_locations else self.z_ancilla
            reset_data = x_initlized_data  if ancilla in x_ancilla_locations else z_initlized_data
            for orig_ancilla in original_ancillas:
                if all((qubit in reset_data) or (self.get_qubit(*(qubit), return_none_if_missing = True) in orig_ancilla.get_support())
                        for qubit in ancilla_support):
                    circuit.append("DETECTOR", get_rec(ancilla, measurement_order), ancilla)
                    break
        
       
        for _ in range(rounds):
            syndrome_cycle(circuit, x_ancilla_locations, z_ancilla_locations, data_qubits, qubit_location_index_dict, measurement_order, 
                           rotated_code=False)
            for qubit in measurement_order:
                circuit.append("DETECTOR", [get_rec(qubit, measurement_order), 
                                            get_rec(qubit, measurement_order, offset = len(measurement_order))], qubit)

        # Final round of measurements
        for ancilla in x_ancilla_locations:
            circuit.append("MPP",
                stim.target_combined_paulis([stim.target_pauli(qubit_location_index_dict[qubit], "X")
                                           for qubit in ancilla_support_in_expanded_code(ancilla, data_qubits, rotated_code=False)]))

        for ancilla in z_ancilla_locations:
            circuit.append("MPP", 
                stim.target_combined_paulis([stim.target_pauli(qubit_location_index_dict[qubit], "Z")
                                           for qubit in ancilla_support_in_expanded_code(ancilla, data_qubits, rotated_code=False)]))

        # Add detectors after all measurements
        for ancilla in x_ancilla_locations:
            circuit.append("DETECTOR", [get_rec(ancilla, measurement_order), get_rec(ancilla, measurement_order, offset=len(measurement_order))], ancilla)

        for ancilla in z_ancilla_locations:
            circuit.append("DETECTOR", [get_rec(ancilla, measurement_order, offset=len(measurement_order)),
                                      get_rec(ancilla, measurement_order)], ancilla)

        # Measure logical Z operator on all data qubits
        middle_qubit = (self.d - 1, self.d - 1)
        middle_qubit_x, middle_qubit_y = rotate(middle_qubit)
        if logical_measurement_basis == "x":
            circuit.append("MPP",
                stim.target_combined_paulis([stim.target_pauli(qubit_location_index_dict[(x,y)], "X")
                                           for x,y in data_qubits if x == 0]))
        elif logical_measurement_basis == "z":
            circuit.append("MPP",
                stim.target_combined_paulis([stim.target_pauli(qubit_location_index_dict[(x,y)], "Z")
                                           for x,y in data_qubits if y == 0]))
        else:
            raise ValueError(f"Invalid logical measurement basis: {logical_measurement_basis}")
        circuit.append("OBSERVABLE_INCLUDE", stim.target_rec(-1), [0])
        return circuit
  
    def get_cirq_logical_x(self):
        return cirq.PauliString({self.cirq_qubits[qubit.index]: cirq.X for qubit in self.logical_x_qubits})

    def get_cirq_logical_z(self):
        return cirq.PauliString({self.cirq_qubits[qubit.index]: cirq.Z for qubit in self.logical_z_qubits})

    def get_cirq_qubit_map(self):
        return {self.cirq_qubits[qubit.index]:qubit.index for qubit in self.data_qubits + self.ghz_ancilla + self.residual_qubits}

    def generate_stim_stabilizer_circuit(self):
        circuit = stim.Circuit()
        for qubit in self.qubits_to_simulate:
            circuit.append("QUBIT_COORDS", qubit.index, (qubit.x, qubit.y))
        
        circuit_string = str(circuit)
        for qubit in self.x_ancilla + self.z_ancilla:
            circuit_string += (f"\n#!pragma MARK{qubit.basis.upper()}({qubit.index}) {' '.join([str(q.index) for q in qubit.get_support()])}")
        
        return circuit_string

    def circuit_from_stim_file(self, filename : str, reverse = False):
        stim_circuit = stim.Circuit.from_file(filename)
        stim_circuit = stim_circuit if not reverse else stim_circuit[::-1]
        circuit = Circuit(self)

        # Extract qubit coordinates from stim circuit
        qubit_coords = {}
        for instruction in stim_circuit:
            if instruction.name == "QUBIT_COORDS":
                qubit_index = instruction.targets_copy()[0]
                coords = instruction.gate_args_copy()
                qubit_coords[qubit_index] = self.get_qubit(*coords)

        for instruction in stim_circuit:
            if instruction.name == "RX":
                for target in instruction.targets_copy():
                    circuit.append(ResetX(self, qubit_coords[target]))
            elif instruction.name == "R":
                for target in instruction.targets_copy():
                    circuit.append(Reset(self, qubit_coords[target]))
            elif instruction.name == "TICK":
                circuit.append(Tick(self))
            elif instruction.name == "CX":
                for idx in range(0,len(instruction.targets_copy()),2):
                    circuit.append(CX(self, qubit_coords[instruction.targets_copy()[idx]], qubit_coords[instruction.targets_copy()[idx+1]],
                                      unitary=0))

        return circuit





class Gate:
    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        self.num_operators = 4**len(self.qubits()) - 1

    @property
    def noise_terms(self):
        return [
            [None,
             self.simulation.x_gate(qubit),
             self.simulation.y_gate(qubit),
             self.simulation.z_gate(qubit),
             ]
            for qubit in self.qubits()
        ]
    def apply_statevector(self, statevector : qp.Qobj):
        return self.unitary * statevector

    def apply_statevector_noise(self, statevector : qp.Qobj, noise : float):
        statevector = self.apply_statevector(statevector)
        if np.random.rand() < noise:
            for qubit in self.qubits():
                error_term = random.choice([None, self.simulation.x_gate,
                               self.simulation.y_gate,
                               self.simulation.z_gate])
                if error_term is not None:
                    statevector = chosen_noise(qubit) * statevector                     
        return statevector

    def apply(self, rho : qp.Qobj):
        return self.unitary * rho * (self.unitary.dag())


    def apply_noise(self, rho : qp.Qobj, noise : float):
        final_rho = self.apply(rho) * (1-noise)


        for noise_operators in product(*self.noise_terms):
            if any(op is not None for op in noise_operators):
                for op in noise_operators:
                    temp_rho = rho
                    if op is not None:
                        temp_rho = op*temp_rho*op
                final_rho += temp_rho * noise / self.num_operators
        return final_rho

    def plot(self):
        self.qubits().plot()
        if len(self.qubits()) == 3:
            q1, q2, q3 = self.qubits()
            plt.plot([q1.x ,(q2.x+q3.x)/2,q2.x, q3.x],
                     [q1.y, (q2.y + q3.y) / 2, q2.y, q3.y],
                     '-o',
                    color = 'black', alpha = 0.3)
        else:
            plt.plot([q.x for q in self.qubits()],
                 [q.y for q in self.qubits()], '-o',
                     color = 'black', alpha = 0.3)
    def __repr__(self):
        return f"<{self.name}, {self.qubits()}>"



class RQubit(Qubit):
    def is_data(self):
        return self.x%2 == 0 and self.y%2 == 0

    def is_ancilla(self):
        return not self.is_data()

    def get_support(self):
        if self.basis == "x":
            return QubitList([
                self.simulation.get_qubit(self.x + direction[0], self.y + direction[1])
                for direction in rotated_order_x
                if self.simulation.get_qubit(self.x + direction[0], self.y + direction[1], return_none_if_missing = True) != None
            ])
        else:
            return QubitList([
                self.simulation.get_qubit(self.x + direction[0], self.y + direction[1])
                for direction in rotated_order_z
                if self.simulation.get_qubit(self.x + direction[0], self.y + direction[1], return_none_if_missing = True ) != None
            ])

    @property
    def basis(self):
        assert self.is_ancilla()
        if (self.x//2 + self.y//2 - 1) % 2 == 0:
            return "z"
        else:
            return "x"
 

class RotatedSurfaceCodeSimulation(Simulation):
    def __init__(self, d):
        self.d = d
        self.ancilla_offset = 2*d + 1
        self.identity = 0

        self.data_qubits : QubitList = QubitList([RQubit(2*x, 2*y, self) for x in range(self.d) for y in range(self.d)])
        self.ghz_ancilla : QubitList[RQubit] = QubitList([
            RQubit(x + self.ancilla_offset, y, self)
            for x in range (2*d - 1)
                for y in range(2*d - 1)
            if (x + y) % 2 == 0 and x <= y and ((x != y) or x % 2 == 0)
        ])

        self.x_ancilla : QubitList = QubitList()
        self.z_ancilla : QubitList = QubitList()

        for x in range(-1, self.d):
            for y in range(-1, self.d):
                if (x + y) % 2 == 0 and x >= 0 and x < self.d - 1:
                    self.x_ancilla.append(RQubit(2*x + 1, 2*y + 1, self))
                elif (x + y) % 2 == 1 and y >= 0 and y < self.d - 1:
                    self.z_ancilla.append(RQubit(2*x + 1, 2*y + 1, self))

        self.residual_qubits : QubitList = QubitList()
        self.extra_qubits = QubitList([]) # Used only for expansion

        self.generate_qubit_lists()
        self.logical_x_qubits = QubitList([qubit for qubit in self.data_qubits if qubit.x == 0 ])
        self.logical_z_qubits = QubitList([qubit for qubit in self.data_qubits if qubit.y == 0 ])
        self.generate_qubit_dict()
    
    def generate_qubit_lists(self):
        self.qubits_to_simulate = self.data_qubits + self.x_ancilla + self.z_ancilla + self.ghz_ancilla + self.extra_qubits
        self.all_qubits = self.data_qubits + self.ghz_ancilla + self.x_ancilla + self.z_ancilla + self.extra_qubits

    def preplot(self):
        plt.xlim(-3, self.ancilla_offset + 2 * self.d)
        plt.ylim(-3, 2 * self.d)
        plt.grid()
    
    def is_corner_qubit(self, qubit : Qubit):
        if self.d == 3:
            if (qubit.x == 0 or qubit.x == 2*self.d - 2) and (qubit.y == 0 or qubit.y == 2*self.d - 2):
                return True
            return False
        elif self.d == 5:
            if ((qubit.x == 0 or qubit.x == 2*self.d - 2) or (qubit.y == 0 or qubit.y == 2*self.d - 2)) and 4 not in [qubit.x, qubit.y]:
                return True
            if qubit.x in [1, 7] and qubit.y in [1, 7]:
                return True
            return False
        else:
            raise Exception(f"Don't know how to check if {qubit} is a corner qubit for d = {self.d}")



    def add_syndrome_step(self, circuit : Circuit, index: int, unitary = None):
        for cycle_index, (x_direction, z_direction) in enumerate(zip(rotated_order_x, rotated_order_z)):
            if cycle_index != index:
                continue
            for ancilla in self.z_ancilla:
                qubit = self.get_qubit(ancilla.x + z_direction[0], ancilla.y + z_direction[1], return_none_if_missing = True)
                if qubit != None:
                    circuit.append(CX(self, qubit, ancilla, unitary = unitary))
            for ancilla in self.x_ancilla:
                qubit = self.get_qubit(ancilla.x + x_direction[0], ancilla.y + x_direction[1], return_none_if_missing = True)
                if qubit != None:
                    circuit.append(CX(self, ancilla, qubit, unitary = unitary))
                    
    def generate_syndrome_circuit(self, detect_x = True, detect_z = True):
        circuit = Circuit(self)
        unitary = 0

        for qubit in self.x_ancilla:
            circuit.append(ResetX(self, qubit))
        for qubit in self.z_ancilla:
            circuit.append(Reset(self, qubit))
        circuit.append(Tick(self))

        for index in range(4):
            self.add_syndrome_step(circuit, index, unitary = unitary)
            circuit.append(Tick(self))

        for qubit in self.x_ancilla:
            circuit.append(H(self, qubit, unitary = unitary))
        circuit.append(Tick(self))

        for qubit in self.x_ancilla + self.z_ancilla:
            circuit.append(MeasureReset(self, qubit, projector = unitary, detect = ((qubit in self.x_ancilla) and detect_x) or ((qubit in self.z_ancilla) and detect_z)))
            
        circuit.append(Tick(self))
        return circuit

    def generate_projection_circuit_HXY(self):
        circuit = Circuit(self)
        unitary = 0

        for qubit in self.x_ancilla + QubitList([self.ghz_ancilla[3]]):
            circuit.append(ResetX(self, qubit))
        for qubit in self.z_ancilla + QubitList([q for q in self.ghz_ancilla if self.ghz_ancilla.index(q) != 3]):
            circuit.append(Reset(self, qubit))
        circuit.append(Tick(self))

        expansion_steps_before_syndrome = self.expansion_length() - 2
        for idx in range(expansion_steps_before_syndrome):
            self.expand_ancilla_step(circuit, idx, unitary)
            circuit.append(Tick(self))

        for idx in [0,1]:
            self.add_syndrome_step(circuit, idx, unitary = unitary)
            self.expand_ancilla_step(circuit, self.expansion_length() - 2 + idx, unitary)
            circuit.append(Tick(self))
        
        self.phase_kickback_HXY(circuit, unitary = unitary)

        for idx in [0, 1]:
            self.add_syndrome_step(circuit, idx + 2, unitary = unitary)
            self.expand_ancilla_step(circuit, self.expansion_length() - 1 - idx, unitary)
            circuit.append(Tick(self))

        for qubit in self.x_ancilla:
            circuit.append(H(self, qubit, unitary = unitary))
        if expansion_steps_before_syndrome > 0:
            self.expand_ancilla_step(circuit, self.expansion_length() - 3, unitary)
        circuit.append(Tick(self))

        for qubit in self.x_ancilla + self.z_ancilla:
            circuit.append(MeasureReset(self, qubit, projector = unitary, detect = True))

        for idx in range(3, self.expansion_length()):
            self.expand_ancilla_step(circuit, self.expansion_length() - 1 - idx, unitary)
            circuit.append(Tick(self))

        circuit.append(H(self, self.ghz_ancilla[3], unitary = unitary))
        circuit.append(Tick(self))
        for qubit in self.ghz_ancilla:
            circuit.append(MeasureReset(self, qubit, projector = unitary, detect = True))
        circuit.append(Tick(self))
        return circuit

    def generate_projection_double_ghz_HXY(self, replace_phase_kickback = False, target_gate : Gate = None):
        circuit = Circuit(self)
        unitary = 0

        for qubit in (self.x_ancilla):
            circuit.append(ResetX(self, qubit))
        for qubit in self.z_ancilla:
            circuit.append(Reset(self, qubit))
        circuit.append(Tick(self))

        self.add_syndrome_step(circuit, 0, unitary = unitary)
        circuit.append(Tick(self))

        self.add_syndrome_step(circuit, 1, unitary = unitary)
        for qubit in self.ghz_ancilla:
            circuit.append(ResetX(self, qubit))
        circuit.append(Tick(self))

        if replace_phase_kickback:
            self.phase_kickback_CZ(circuit, unitary = unitary, target_gate = target_gate, label = False)
        else:   
            self.phase_kickback_HXY(circuit, unitary = unitary)

        self.collapse_ancilla_for_circuit(circuit, unitary = unitary)
        
        circuit.append(MeasureResetX(self, self.ghz_ancilla[3], projector=unitary, detect=True))
        circuit.append(Tick(self))

        self.expand_ancilla_for_circuit(circuit, unitary = unitary)
        if replace_phase_kickback:
            self.phase_kickback_CZ(circuit, unitary = unitary, target_gate = target_gate, label = False)
        else:
            self.phase_kickback_HXY(circuit, unitary = unitary)

        for qubit in self.ghz_ancilla:
            circuit.append(MeasureResetX(self, qubit, projector = unitary, detect = True))
        self.add_syndrome_step(circuit, 2, unitary = unitary)
        circuit.append(Tick(self))

        self.add_syndrome_step(circuit, 3, unitary = unitary)
        circuit.append(Tick(self))

        for qubit in self.x_ancilla:
            circuit.append(MeasureResetX(self, qubit, projector = unitary, detect = True))
        for qubit in self.z_ancilla:
            circuit.append(MeasureReset(self, qubit, projector = unitary, detect = True))
        circuit.append(Tick(self))
        return circuit

    def stim_expansion_circuit(self, d2: int, rounds : int, logical_measurement_basis : str = "z"):
        circuit = stim.Circuit()
        circuit.append("TICK")
        logical_measurement_basis = logical_measurement_basis.lower()

        x_ancilla_locations = []
        z_ancilla_locations = []

        for x in range(-1, d2):
            for y in range(-1, d2):
                if (x + y) % 2 == 0 and x >= 0 and x < d2 - 1:
                    x_ancilla_locations.append((2*x + 1, 2*y + 1))
                elif (x + y) % 2 == 1 and y >= 0 and y < d2 - 1:
                    z_ancilla_locations.append((2*x + 1, 2*y + 1))
        
        data_qubits = []

        for x in range(d2):
            for y in range(d2):
                data_qubits.append((2*x, 2*y))

        qubit_location_index_dict = {}
        for qubit in self.x_ancilla + self.z_ancilla + self.data_qubits + self.ghz_ancilla + self.extra_qubits + self.residual_qubits:
            qubit_location_index_dict[(qubit.x, qubit.y)] = qubit.index


        current_index = max(qubit_location_index_dict.values())*self.d**2 + 1

        x_initlized_data = []
        z_initlized_data = []
        measurement_order = x_ancilla_locations + z_ancilla_locations 

        for qubit in x_ancilla_locations + z_ancilla_locations + data_qubits :
            if qubit not in qubit_location_index_dict:
                qubit_location_index_dict[qubit] = current_index
                circuit.append("QUBIT_COORDS", current_index, qubit)
                current_index += 1

        for qubit in x_ancilla_locations:
            circuit.append("RX", qubit_location_index_dict[qubit])

        for qubit in z_ancilla_locations:
            circuit.append("R", qubit_location_index_dict[qubit])
            
        for qubit in data_qubits:
            x, y = qubit
            if x > 2*self.d - 2 or y > 2*self.d - 2:
                if x >= y:
                    circuit.append("R", qubit_location_index_dict[qubit])
                    z_initlized_data.append(qubit)
                elif x < y:
                    circuit.append("RX", qubit_location_index_dict[qubit])
                    x_initlized_data.append(qubit)
        
        circuit.append("TICK")

        syndrome_cycle(circuit, x_ancilla_locations, z_ancilla_locations, data_qubits, qubit_location_index_dict, measurement_order)

        for ancilla in x_ancilla_locations:
            ancilla_support = ancilla_support_in_expanded_code(ancilla, data_qubits)
            if any(all((qubit in x_initlized_data) or (self.get_qubit(qubit[0], qubit[1], return_none_if_missing = True) is not None)
                   for qubit in ancilla_support) for orig_ancilla in self.x_ancilla):
                circuit.append("DETECTOR", get_rec(ancilla, measurement_order), ancilla)
        
        for ancilla in z_ancilla_locations:
            ancilla_support = ancilla_support_in_expanded_code(ancilla, data_qubits)
            if all((qubit in z_initlized_data) or (self.get_qubit(qubit[0], qubit[1], return_none_if_missing = True) is not None)
                   for qubit in ancilla_support):
                circuit.append("DETECTOR", get_rec(ancilla, measurement_order), ancilla)
        
        for _ in range(rounds):
            syndrome_cycle(circuit, x_ancilla_locations, z_ancilla_locations, data_qubits, qubit_location_index_dict, measurement_order)
            for qubit in measurement_order:
                circuit.append("DETECTOR", [get_rec(qubit, measurement_order), 
                                            get_rec(qubit, measurement_order, offset = len(measurement_order))], qubit)

        # Final round of measurements
        for ancilla in x_ancilla_locations:
            circuit.append("MPP",
                stim.target_combined_paulis([stim.target_pauli(qubit_location_index_dict[qubit], "X")
                                           for qubit in ancilla_support_in_expanded_code(ancilla, data_qubits)]))

        for ancilla in z_ancilla_locations:
            circuit.append("MPP", 
                stim.target_combined_paulis([stim.target_pauli(qubit_location_index_dict[qubit], "Z")
                                           for qubit in ancilla_support_in_expanded_code(ancilla, data_qubits)]))

        # Add detectors after all measurements
        for ancilla in x_ancilla_locations:
            circuit.append("DETECTOR", [get_rec(ancilla, measurement_order), get_rec(ancilla, measurement_order, offset=len(measurement_order))], ancilla)

        for ancilla in z_ancilla_locations:
            circuit.append("DETECTOR", [get_rec(ancilla, measurement_order, offset=len(measurement_order)),
                                      get_rec(ancilla, measurement_order)], ancilla)

        # Measure logical Z operator on all data qubits
        
        if logical_measurement_basis == "z":
            circuit.append("MPP",
                stim.target_combined_paulis([stim.target_pauli(qubit_location_index_dict[(x,y)], "Z")
                                           for x,y in data_qubits if y == 0]))
        elif logical_measurement_basis == "x":
            circuit.append("MPP",
                stim.target_combined_paulis([stim.target_pauli(qubit_location_index_dict[(x,y)], "X")
                                           for x,y in data_qubits if x == 0]))
        else:
            raise ValueError(f"Invalid logical measurement basis: {logical_measurement_basis}")
        circuit.append("OBSERVABLE_INCLUDE", stim.target_rec(-1), [0])
        return circuit
            
            
       
class Label(Gate):
    def apply(self, rho : qp.Qobj):
        return rho

    def apply_statevector(self, statevector : qp.Qobj):
        return statevector

    def apply_statevector_noise(self, statevector : qp.Qobj, noise : float):
        return statevector

    def qubits(self):
        return QubitList([])


class BeginPhaseKickback(Label):
    name = 'BeginPhaseKickback'

class Tick(Label):
    name = 'Tick'

class CX(Gate):
    def __init__(self, simulation: Simulation, control: Qubit, target: Qubit, unitary = None):
        self.control = control
        self.target = target
        self.name = 'CX'
        Gate.__init__(self, simulation)

    def qubits(self):
        return QubitList([self.control, self.target])

class CZ(Gate):
    def __init__(self, simulation: Simulation, control: Qubit, target: Qubit, unitary = None):
        self.control = control
        self.target = target
        self.name = 'CZ'
        Gate.__init__(self, simulation) 

    def qubits(self):
        return QubitList([self.control, self.target])

class S(Gate):
    def __init__(self, simulation: Simulation, qubit: Qubit, unitary = None):
        self.qubit = qubit
        self.name = 'S'
        Gate.__init__(self, simulation)
    def qubits(self):
        return QubitList([self.qubit])

class Sdag(Gate):
    def __init__(self, simulation: Simulation, qubit: Qubit, unitary = None):
        self.qubit = qubit
        self.name = 'Sdag'
        Gate.__init__(self, simulation)
        
    def qubits(self):
        return QubitList([self.qubit])

class T(Gate):
    def __init__(self, simulation: Simulation, qubit: Qubit, unitary = None):
        self.qubit = qubit
        self.name = 'T'
        Gate.__init__(self, simulation)
    
    def qubits(self):
        return QubitList([self.qubit])

class Tdag(Gate):
    def __init__(self, simulation: Simulation, qubit: Qubit, unitary = None):
        self.qubit = qubit
        self.name = 'Tdag'
        Gate.__init__(self, simulation)
    def qubits(self):
        return QubitList([self.qubit])

class CSwap(Gate):
    def __init__(self, simulation: Simulation, control: Qubit, target_a: Qubit, target_b: Qubit, unitary = None):
        self.control = control
        self.target_a = target_a
        self.target_b = target_b
        self.name = 'CSwap'
        Gate.__init__(self, simulation)

    def qubits(self):
        return QubitList([self.control,self.target_a, self.target_b])

class CH(Gate):
    def __init__(self, simulation: Simulation, control: Qubit, target: Qubit, unitary = None):
        self.control = control
        self.target = target
        self.name = 'CH'
        Gate.__init__(self, simulation)

    def qubits(self):
        return QubitList([self.control, self.target])

class CSX(Gate):
    def __init__(self, simulation: Simulation, control: Qubit, target: Qubit, unitary = None):
        self.control = control
        self.target = target
        self.name = 'CSX'
        Gate.__init__(self, simulation)

    def qubits(self):
        return QubitList([self.control, self.target])

class CS_dagX(Gate):
    def __init__(self, simulation: Simulation, control: Qubit, target: Qubit, unitary = None):
        self.control = control
        self.target = target
        self.name = 'CS_DAGX'
        Gate.__init__(self, simulation)

    def qubits(self):
        return QubitList([self.control, self.target])

class CXS(Gate):
    def __init__(self, simulation: Simulation, control: Qubit, target: Qubit, unitary = None):
        self.control = control
        self.target = target
        self.name = 'CXS'
        Gate.__init__(self, simulation)
    
    def qubits(self):
        return QubitList([self.control, self.target])

class CXS_DAG(Gate):
    def __init__(self, simulation: Simulation, control: Qubit, target: Qubit, unitary = None):
        self.control = control
        self.target = target
        self.name = 'CXS_DAG'
        Gate.__init__(self, simulation)
        
    def qubits(self):
        return QubitList([self.control, self.target])
        

class H(Gate):
    def __init__(self, simulation: Simulation, qubit: Qubit, unitary = None):
        self.qubit = qubit
        self.name = 'H'
        Gate.__init__(self, simulation)

    def qubits(self):
        return QubitList([self.qubit])

class MeasureReset(Gate):
    def __init__(self, simulation: Simulation, qubit: Qubit, projector = None, detect = True):
        self.qubit = qubit
        self.name = 'MeasureReset'
        self.detect = detect
        Gate.__init__(self, simulation)

    def qubits(self):
        return QubitList([self.qubit])

class MeasureResetX(Gate):
    def __init__(self, simulation: Simulation, qubit: Qubit, projector = None, detect = True):
        self.qubit = qubit
        self.name = 'MeasureResetX'
        self.detect = detect
        Gate.__init__(self,simulation)
    def qubits(self):
        return QubitList([self.qubit])
    
class CCZ(Gate):
    def __init__(self, simulation: Simulation, control_a: Qubit, control_b: Qubit, target: Qubit, unitary = None):
        self.control_a = control_a
        self.control_b = control_b
        self.target = target
        self.name = 'CCZ'
        Gate.__init__(self, simulation)

    def qubits(self):
        return QubitList([self.control_a, self.control_b, self.target])

class ResetX(Gate):
    def __init__(self, simulation: Simulation, qubit: Qubit):
        self.qubit = qubit
        self.name = 'ResetX'
        Gate.__init__(self, simulation)

    def qubits(self):
        return QubitList([self.qubit])

       

class Reset(Gate):
    def __init__(self, simulation: Simulation, qubit: Qubit):
        self.qubit = qubit
        self.name = 'Reset'
        Gate.__init__(self, simulation)

    def qubits(self):
        return QubitList([self.qubit])

class Circuit:
    def __init__(self, simulation: Simulation, gates : list[Gate] = None):
        self.simulation : Simulation = simulation
        if gates is None:
            self.gates: list[Gate]= []
        else:
            self.gates: list[Gate]= gates

    def __len__(self):
        tick_index = 1
        for gate in self.gates:
            if isinstance(gate, Tick):
                tick_index += 1
        return tick_index

    def append(self, gate : Gate):
        self.gates.append(gate)

    def plot(self, tick_index = 0):
        self.simulation.preplot()
        for gate in self.gates:
            if isinstance(gate,Tick):
                tick_index -= 1
                continue

            if tick_index == 0:
                gate.plot()
    
    def expand_simulation(self, new_distance : int):

        simulation_class = type(self.simulation)
        new_simulation = simulation_class(d = new_distance)
        for qubit in self.simulation.all_qubits:
            if new_simulation.get_qubit(qubit.x, qubit.y, return_none_if_missing = True) is None:
                new_qubit = Qubit(qubit.x, qubit.y, new_simulation)
                new_simulation.extra_qubits.append(new_qubit)
        new_simulation.generate_qubit_lists()
        new_simulation.generate_qubit_dict()

        new_circuit = Circuit(new_simulation)
        for gate in self.gates:
            gate_type = type(gate)
            new_qubits = [new_simulation.get_qubit(qubit.x, qubit.y) for qubit in gate.qubits()]

            if gate_type in [MeasureReset, MeasureResetX]:
                new_circuit.append(gate_type(new_simulation, *new_qubits, detect = gate.detect))
            elif gate_type in [Reset]:
                new_circuit.append(gate_type(new_simulation, *new_qubits))
            elif gate_type in [ResetX]:
                new_circuit.append(gate_type(new_simulation, *new_qubits))
            elif gate_type in [Tick, BeginPhaseKickback]:
                new_circuit.append(gate_type(new_simulation))
            else:
                new_circuit.append(gate_type(new_simulation, *new_qubits))
        return new_simulation, new_circuit




    def plot_timeline(self, lines = None, figsize = (20, 20)):
        plt.figure(figsize = figsize)
        count_ticks = sum(1 for i in self.gates if isinstance(i,Tick))
        max_x, min_x = max(qubit.x for qubit in self.simulation.all_qubits), min(qubit.x for qubit in self.simulation.all_qubits)
        max_y, min_y = max(qubit.y for qubit in self.simulation.all_qubits), min(qubit.y for qubit in self.simulation.all_qubits)

        x_lines = lines if lines is not None else int(round(count_ticks**0.5))
        y_lines = count_ticks//x_lines+1
        for i in range(count_ticks):
            plt.xlim(min_x-1, max_x+1)
            plt.ylim(min_y-1, max_y+1)
            plt.subplot(y_lines,x_lines, i+1)
            self.simulation.preplot()
            plt.grid()
            self.simulation.plot()
            self.plot(i)
    
    def __add__(self, other : Circuit):
        assert self.simulation == other.simulation
        return Circuit(self.simulation, self.gates + other.gates)

    def to_qiskit_circuit(self, initial_state : np.array = None):
        count_measurements = sum(1 for i in self.gates if isinstance(i, MeasureReset))
        qiskit_circuit = qk.QuantumCircuit(qk.QuantumRegister(self.simulation.count_simulated_qubits()), 
                                           qk.ClassicalRegister(count_measurements))
        if initial_state is not None:
            qiskit_circuit.initialize(initial_state)
            return qiskit_circuit

        measurement_index = 0
        for gate in self.gates:
            if isinstance(gate, H):
                qiskit_circuit.h(gate.qubit.index)
            elif isinstance(gate, CX):
                qiskit_circuit.cx(gate.control.index, gate.target.index)
            elif isinstance(gate, CSwap):
                qiskit_circuit.ccx(gate.control.index, gate.target_a.index, gate.target_b.index)
                qiskit_circuit.ccx(gate.control.index, gate.target_b.index, gate.target_a.index)
                qiskit_circuit.ccx(gate.control.index, gate.target_a.index, gate.target_b.index)
            elif isinstance(gate, CH):
                qiskit_circuit.ch(gate.control.index, gate.target.index)
            elif isinstance(gate, MeasureReset):
                qiskit_circuit.measure(gate.qubit.index, measurement_index)
                qiskit_circuit.reset(gate.qubit.index)
                measurement_index += 1
        return qiskit_circuit
    
    def to_cirq_circuit(self, noise : float = 0, remove_measurements = False):
        count_measurements = sum(1 for i in self.gates if isinstance(i, MeasureReset))

        cirq_circuit = cirq.Circuit()
        qubits = self.simulation.cirq_qubits

        measurement_index = 0
        for gate in self.gates:
            if isinstance(gate, H):
                cirq_circuit.append(cirq.H(qubits[gate.qubit.index]))
                if noise != 0:
                    cirq_circuit.append(cirq.depolarize(noise).on(qubits[gate.qubit.index]))

            elif isinstance(gate, CX):
                cirq_circuit.append(cirq.CX(qubits[gate.control.index], qubits[gate.target.index]))
                if noise != 0:
                    cirq_circuit.append(cirq.depolarize(noise, 2).on(qubits[gate.control.index], qubits[gate.target.index]))

            elif isinstance(gate, CSwap):
                cirq_circuit.append(cirq.CSWAP(qubits[gate.control.index], qubits[gate.target_a.index], qubits[gate.target_b.index]))
                if noise != 0:
                    cirq_circuit.append(cirq.depolarize(noise, 3).on(qubits[gate.control.index], qubits[gate.target_a.index], qubits[gate.target_b.index]))

            elif isinstance(gate, CH):

                cirq_circuit.append(cirq.Ry(rads = np.pi/4).on(qubits[gate.target.index]))
                cirq_circuit.append(cirq.CZ(qubits[gate.control.index], qubits[gate.target.index]))
                cirq_circuit.append(cirq.Ry(rads = -np.pi/4).on(qubits[gate.target.index]))
                if noise != 0:
                    cirq_circuit.append(cirq.depolarize(noise, 2).on(qubits[gate.control.index], qubits[gate.target.index]))

            elif isinstance(gate, MeasureReset) and not remove_measurements:
                cirq_circuit.append(cirq.measure(qubits[gate.qubit.index], key = f"m_{measurement_index}"))
                measurement_index += 1
        
        return cirq_circuit
    
    def to_pennylane_circuit(self):
        dev = qml.device("default.qubit", wires = self.simulation.count_simulated_qubits())
        print(self.simulation.count_simulated_qubits())
        @qml.qnode(dev)
        def circuit():
            for gate in self.gates:
                if isinstance(gate, H):
                    qml.Hadamard(wires = gate.qubit.index)
                elif isinstance(gate, CX):
                    qml.CNOT(wires = [gate.control.index, gate.target.index])
                elif isinstance(gate, CSwap):
                    qml.CSWAP(wires = [gate.control.index, gate.target_a.index, gate.target_b.index])
                elif isinstance(gate, CH):
                    qml.ControlledQubitUnitary(qml.Hadamard(gate.target.index), control_wires = gate.control.index)
                elif isinstance(gate, MeasureReset):
                    qml.measure(gate.qubit.index, postselect=0,reset=True)
        
            x_operator = reduce(lambda x,y: x@y, [qml.X(q.index) for q in self.simulation.logical_x_qubits])
            z_operator = reduce(lambda x,y: x@y, [qml.Z(q.index) for q in self.simulation.logical_z_qubits])
            return qml.expval(x_operator), qml.expval(z_operator),qml.state()

        return circuit
    def to_stim_circuit(self, noise_model : NoiseModel | None = None, p: float = 0, apply_non_cliffords = True, 
                        post_select_syndromes = True, measure_logical_operator = True,
                        non_clifford_noise_strategy = "NOISE", logical_measurement_basis : str = "z"):
        # Convert circuit to stim format
        stim_circuit = stim.Circuit()
        logical_measurement_basis = logical_measurement_basis.lower()
        previous_gate = None

                # Add qubit coordinates based on syndrome locations
        coords = {}

        for q in self.simulation.qubits_to_simulate:
            coords[q.index] = (q.x, q.y)

        # Add coordinates to stim circuit
        for q in coords:
            stim_circuit.append("QUBIT_COORDS", q, coords[q])

        # Convert gates to stim format
        for gate in self.gates:
            if isinstance(gate, H):
                stim_circuit.append("H", [gate.qubit.index])
            elif isinstance(gate, Tick):
                if previous_gate is not None and not isinstance(previous_gate, Tick):
                    stim_circuit.append("TICK")
            elif isinstance(gate, CX):
                stim_circuit.append("CX", [gate.control.index, gate.target.index])
            elif isinstance(gate, CZ):
                stim_circuit.append("CZ", [gate.control.index, gate.target.index])
            elif isinstance(gate, CSwap):
                if apply_non_cliffords:
                    stim_circuit.append(convertion_dict['CSWAP'], [gate.control.index, gate.target_a.index, gate.target_b.index])
                if p != 0:
                    three_qubit_correlated_noise(stim_circuit, p, gate)
            elif isinstance(gate, CCZ):
                if apply_non_cliffords:
                    stim_circuit.append(convertion_dict['CCZ'], [gate.control_a.index, gate.control_b.index, gate.target.index])
                if p != 0:
                    three_qubit_correlated_noise(stim_circuit, p, gate)
            elif isinstance(gate, CH):
                if apply_non_cliffords:
                    #XCX is a two qubit operation and therefore the noise model applies 2Q noise correctly
                    stim_circuit.append(convertion_dict['CH'], [gate.control.index, gate.target.index])
                elif non_clifford_noise_strategy == "NOISE":
                    #If we don't apply the CH, we need to apply 2Q noise manually
                    stim_circuit.append("DEPOLARIZE2", [gate.control.index, gate.target.index], p)
                else:
                    add_gate_using_non_clifford_strategy(stim_circuit, gate, non_clifford_noise_strategy, p)
            elif isinstance(gate, CSX) or isinstance(gate, CS_dagX):
                if apply_non_cliffords:
                    stim_circuit.append(convertion_dict[gate.name], [gate.control.index, gate.target.index])
                elif non_clifford_noise_strategy == "NOISE":
                    stim_circuit.append("DEPOLARIZE2", [gate.control.index, gate.target.index], p)
                else:
                    add_gate_using_non_clifford_strategy(stim_circuit, gate, non_clifford_noise_strategy, p)
            elif isinstance(gate, MeasureReset):
                stim_circuit.append("MR", [gate.qubit.index])
                if gate.detect:
                    stim_circuit.append("DETECTOR", stim.target_rec(-1))
                else:
                    fix_qubits = gate.qubit.get_correction()
                    for fix_qubit in fix_qubits:
                        if gate.qubit.basis == "x":
                            stim_circuit.append("CZ", [stim.target_rec(-1), fix_qubit.index])
                        elif gate.qubit.basis == "z":
                            stim_circuit.append("CX", [stim.target_rec(-1), fix_qubit.index])
            elif isinstance(gate, MeasureResetX):
                stim_circuit.append("MRX", [gate.qubit.index])
                if gate.detect:
                    stim_circuit.append("DETECTOR", stim.target_rec(-1))
            elif isinstance(gate, Reset):
                stim_circuit.append("R", [gate.qubit.index])
            elif isinstance(gate, ResetX):
                stim_circuit.append("RX", [gate.qubit.index])
            elif isinstance(gate, T):
                if apply_non_cliffords:
                    stim_circuit.append(convertion_dict['T'], [gate.qubit.index])
                elif non_clifford_noise_strategy == "NOISE":
                    stim_circuit.append("DEPOLARIZE1", [gate.qubit.index], p)
            elif isinstance(gate, Tdag):
                if apply_non_cliffords:
                    stim_circuit.append(convertion_dict['T_DAG'], [gate.qubit.index])
                elif non_clifford_noise_strategy == "NOISE":
                    stim_circuit.append("DEPOLARIZE1", [gate.qubit.index], p)
            elif isinstance(gate, S):
                if apply_non_cliffords:
                    stim_circuit.append("S", [gate.qubit.index])
                elif non_clifford_noise_strategy == "NOISE":
                    stim_circuit.append("DEPOLARIZE1", [gate.qubit.index], p)
                else:
                    add_gate_using_non_clifford_strategy(stim_circuit, gate, non_clifford_noise_strategy, p)
            elif isinstance(gate, Sdag):
                if apply_non_cliffords:
                    stim_circuit.append("S_DAG", [gate.qubit.index])
                elif non_clifford_noise_strategy == "NOISE":
                    stim_circuit.append("DEPOLARIZE1", [gate.qubit.index], p)
                else:
                    add_gate_using_non_clifford_strategy(stim_circuit, gate, non_clifford_noise_strategy, p)
            elif isinstance(gate, BeginPhaseKickback):
                if not apply_non_cliffords:
                    add_gate_using_non_clifford_strategy(stim_circuit, gate, non_clifford_noise_strategy, p)
            previous_gate = gate

        if post_select_syndromes:
            for ancilla in self.simulation.x_ancilla + self.simulation.z_ancilla:
                stim_circuit.append("MPP",
                    stim.target_combined_paulis([stim.target_pauli(qubit.index, ancilla.basis)
                                                                for qubit in ancilla.get_support()]))
            for index in range(len(self.simulation.x_ancilla + self.simulation.z_ancilla)):
                stim_circuit.append("DETECTOR",stim.target_rec(-1-index))

        if measure_logical_operator:
            if apply_non_cliffords:
                final_measurement_basis = 'Y'
                logical_operator = self.simulation.logical_z_qubits
            elif logical_measurement_basis == "x":
                final_measurement_basis = 'X'
                logical_operator = self.simulation.logical_x_qubits
            elif logical_measurement_basis == "z":
                final_measurement_basis = 'Z'
                logical_operator = self.simulation.logical_z_qubits
            else:
                raise ValueError(f"Invalid logical measurement basis: {logical_measurement_basis}")
            stim_circuit.append("MPP",
            stim.target_combined_paulis([stim.target_pauli(qubit.index, final_measurement_basis)
                                                              for qubit in logical_operator]))
            stim_circuit.append("OBSERVABLE_INCLUDE", stim.target_rec(-1),[0])

        if noise_model is not None:
            stim_circuit = noise_model.noisy_circuit_skipping_mpp_boundaries(stim_circuit)
        return stim_circuit

def _get_mid(ls):
    return ls[int(len(ls)//2)]

def add_gate_using_non_clifford_strategy(stim_circuit, gate, non_clifford_noise_strategy, p):
    if non_clifford_noise_strategy == "CZ":
        """
        Replaces CH with CZ for non-corner qubits which measures the logical Z operator
        """
        if gate.name == "CH":
            if gate.simulation.is_corner_qubit(gate.target):
                stim_circuit.append("DEPOLARIZE2", [gate.control.index, gate.target.index], p)
            else:
                stim_circuit.append("CZ", [gate.control.index, gate.target.index])
        elif isinstance(gate, T) or isinstance(gate, Tdag):
            stim_circuit.append("DEPOLARIZE1", [gate.qubit.index], p)
        elif isinstance(gate, S):
            stim_circuit.append("S", [gate.qubit.index])
        elif isinstance(gate, Sdag):
            stim_circuit.append("S_DAG", [gate.qubit.index])
        elif isinstance(gate, BeginPhaseKickback):
            pass
        else:
            raise Exception(f"Don't know how to handle gate {gate.name} with non-clifford noise strategy {non_clifford_noise_strategy}")
    elif non_clifford_noise_strategy in ["CZ_HXY", "CX_HXY"]:
        if isinstance(gate, BeginPhaseKickback):
            temp_circuit = Circuit(gate.simulation)
            gate.simulation.phase_kickback_CZ(temp_circuit, unitary = 1, target_gate = (CZ if non_clifford_noise_strategy == "CZ_HXY" else CX))
            for gate in temp_circuit.gates:
                if isinstance(gate, CZ):
                    stim_circuit.append("CZ", [gate.control.index, gate.target.index])
                elif isinstance(gate, CX):
                    stim_circuit.append("CX", [gate.control.index, gate.target.index])
                elif isinstance(gate, Tick):
                    stim_circuit.append("TICK")
                elif isinstance(gate, BeginPhaseKickback):
                    pass
                else:
                    raise Exception(f"Don't know how to handle gate {gate.name} while added CZ measurement")
        elif isinstance(gate, CCZ):
            three_qubit_correlated_noise(stim_circuit, p, gate)
        elif isinstance(gate, CSX) or isinstance(gate, CS_dagX):
            pass
        elif isinstance(gate, Sdag) or isinstance(gate, S) or isinstance(gate, Tdag) or isinstance(gate, T):
            stim_circuit.append("DEPOLARIZE1", [gate.qubit.index], p)
        else:
            raise Exception(f"Don't know how to handle gate {gate.name} with non-clifford noise strategy {non_clifford_noise_strategy}")
    


def three_qubit_correlated_noise(stim_circuit : stim.Circuit, p : float, gate : Gate):
    for paulis in product(*["IXYZ"]*3):
        pauli_noise = [stim.target_pauli(q.index, p)
                                        for p,q in zip(paulis, gate.qubits())
                                            if p != 'I']
        if pauli_noise:
            stim_circuit.append("CORRELATED_ERROR",pauli_noise, 3*p/(4**3-1))

def ancilla_support_in_expanded_code(qubit : tuple[int, int], expanded_data_qubits : list[tuple[int, int]], rotated_code : bool = True):
    support = []
    for dx, dy in rotated_order_x if rotated_code else unrotated_order_x:
        if (qubit[0] + dx, qubit[1] + dy) in expanded_data_qubits:
            support.append((qubit[0] + dx, qubit[1] + dy))
    return support

def get_rec(ancilla : tuple[int, int], measurement_order : list[tuple[int, int]], offset = 0):
    if ancilla in measurement_order:
        return stim.target_rec(-(len(measurement_order)-measurement_order.index(ancilla))-offset)
    else:
        raise Exception(f"Ancilla {ancilla=} not found in measurement order")

def syndrome_cycle(circuit : stim.Circuit, 
                   x_ancilla_locations : list[tuple[int, int]], 
                   z_ancilla_locations : list[tuple[int, int]], 
                   data_qubits : list[tuple[int, int]],
                   qubit_location_index_dict : dict[tuple[int, int], int],
                   measurement_order : list[tuple[int, int]],
                   rotated_code : bool = True):
    for idx in range(4):
        if rotated_code:
            x_direction = rotated_order_x[idx]
            z_direction = rotated_order_z[idx]
        else:
            x_direction = unrotated_order_x[idx]
            z_direction = unrotated_order_z[idx]
        for ancilla in x_ancilla_locations:
            target = (ancilla[0] + x_direction[0], ancilla[1] + x_direction[1])
            if target in qubit_location_index_dict and target in data_qubits:
                circuit.append("CX", [qubit_location_index_dict[target], qubit_location_index_dict[ancilla]][::-1])

        for ancilla in z_ancilla_locations:
            target = (ancilla[0] + z_direction[0], ancilla[1] + z_direction[1])
            if target in qubit_location_index_dict and target in data_qubits:
                circuit.append("CX", [qubit_location_index_dict[ancilla], qubit_location_index_dict[target]][::-1])
        circuit.append("TICK")

    for ancilla in measurement_order:
        if ancilla in x_ancilla_locations:
            circuit.append("MRX", qubit_location_index_dict[ancilla])
        elif ancilla in z_ancilla_locations:
            circuit.append("MR", qubit_location_index_dict[ancilla])
        else:
            raise Exception(f"Ancilla {ancilla=} not found in x_ancilla_locations or z_ancilla_locations")
    circuit.append("TICK")

        


def rotate(coordinates : tuple[int,int]):
    """
    The transforamtion of an unrotated surface code to a rotated surface code
    is defined using the following transformation:
    (2, 0) -> (0, 0)
    (1, 1) -> (0, 2)

    As a function, this is clearly:
    (x, y) -> ((x+y-2)/2, (x-y+2)/2)
    """
    x, y = coordinates
    x_shift = (x - 2)
    return (x_shift + y, y - x_shift)

def rotate_back(coordinates : tuple[int, int]):
    """
    The inverse transformation of rotate
    """
    x, y = coordinates
    x, y = (x - y)//2, (x + y )//2
    x_shift = x + 2
    return (x_shift, y)

def main():
    """rotation tests """
    # Test some known rotations
    assert rotate((2,0)) == (0,0)
    assert rotate((1,1)) == (0,2)
    assert rotate((3,1)) == (2,0)
    assert rotate((2,2)) == (2,2)
    
    # Test round trip rotations
    test_coords = [(2,0), (4,0), (3,1), (2,2), (4,2), (3,3)]
    for coord in test_coords:
        assert rotate_back(rotate(coord)) == coord, (coord, rotate(coord), rotate_back(rotate(coord)))
        assert sum(rotate(coord) )% 2 == 0, (coord, rotate(coord))
        
    # Test that rotated coordinates maintain even parity
    for x in range(6):
        for y in range(6):
            if (x + y) % 2 == 0:  # Only test valid qubit locations
                rotated = rotate((x,y))
                assert (rotated[0] + rotated[1]) % 2 == 0, f"Rotation of {(x,y)} to {rotated} breaks even parity"
    
    print("All rotation tests passed!")

if __name__ == "__main__":
    main()