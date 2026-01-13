import stim
import numpy as np
import sinter
from typing import List, Iterator

# Pauli Gates
PAULI_GATES = ["I", "X", "Y", "Z"]

# Single Qubit Clifford Gates
SINGLE_QUBIT_CLIFFORD_GATES = [
    "C_XYZ", "C_ZYX", "H", "H_XY", "H_XZ", "H_YZ", 
    "S", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", 
    "SQRT_Z", "SQRT_Z_DAG", "S_DAG"
]

SINGLE_QUBIT_GATES = PAULI_GATES + SINGLE_QUBIT_CLIFFORD_GATES

COLLAPSING_GATES = [
    "M", "MR", "MRX", "MRY", "MRZ", 
    "MX", "MY", "MZ", "R", "RX", "RY", "RZ"
]

MEASUREMENTS = [
    "M", "MR", "MRX", "MRY", "MRZ", 
    "MX", "MY", "MZ",
]

# Two Qubit Clifford Gates
TWO_QUBIT_CLIFFORD_GATES = [
    "CNOT", "CX", "CXSWAP", "CY", "CZ", "CZSWAP", 
    "ISWAP", "ISWAP_DAG", "SQRT_XX", "SQRT_XX_DAG", 
    "SQRT_YY", "SQRT_YY_DAG", "SQRT_ZZ", "SQRT_ZZ_DAG", 
    "SWAP", "SWAPCX", "SWAPCZ", "XCX", "XCY", "XCZ", 
    "YCX", "YCY", "YCZ", "ZCX", "ZCY", "ZCZ"
]

SINGLE_QUBIT_ERRORS = ["DEPOLARIZE1", "X_ERROR", "Y_ERROR", "Z_ERROR"]
TWO_QUBIT_ERRORS = ["DEPOLARIZE2"]

OTHER_GATES = ["TICK", "SHIFT_COORDS", "REPEAT", "REPEAT_END", "QUBIT_COORDS", "OBSERVABLE_INCLUDE", "DETECTOR","}"]

NEW_CYCLE_STRING = "SHIFT_COORDS(0, 0, 1)"

def get_erased_circuit(circuit: stim.Circuit, e: float, e1: float = 0, e_SPAM = None, r=0) -> stim.Circuit:
    """
    e is erasure probability per operation
    """

    if e_SPAM is None:
        e_SPAM = e1
    
    erasures = np.zeros(circuit.num_qubits)
    circuit_list = str(circuit).split("\n")
    new_circuit = ""

    for i, op in enumerate(circuit_list):
        if NEW_CYCLE_STRING in op:
            r-=1
        gate = op.lstrip().split(" ")[0].split("(")[0]
        if gate in SINGLE_QUBIT_ERRORS + TWO_QUBIT_ERRORS + MEASUREMENTS and r <= 0: # resets have depolorizing gate while measurements have built-in errors - M(p)
            for qubit in op.split(" "):
                if qubit.isdigit():
                    if gate in TWO_QUBIT_ERRORS:
                        e_gate = e
                    elif gate in ['DEPOLARIZE1']:
                        e_gate = e1
                    else:
                        e_gate = e_SPAM
                    if np.random.rand() < e_gate:
                        erasures[int(qubit)] = 1
            
            if gate in SINGLE_QUBIT_ERRORS or (gate in MEASUREMENTS and _is_measurement_noisy(op)):
                for qubit, erased in enumerate(erasures):
                    if erased > 0:
                        new_circuit += f"DEPOLARIZE1(0.74999) {int(qubit)}\n"
            elif gate in TWO_QUBIT_ERRORS:
                for qubit, erased in enumerate(erasures):
                    if erased > 0:
                        qubits = op.split()[1:]
                        interacting_qubit = [qubits[j+1] if j % 2 == 0 else qubits[j-1] for j in range(len(qubits)) if int(qubits[j]) == qubit][0]
                        new_circuit += f"DEPOLARIZE2(0.93749) {int(qubit)} {int(interacting_qubit)}\n"
            
            new_circuit += op + "\n"

            # reset erasures
            erasures = np.zeros(circuit.num_qubits)
        elif gate in SINGLE_QUBIT_GATES + TWO_QUBIT_CLIFFORD_GATES + COLLAPSING_GATES + OTHER_GATES or r > 0:
            new_circuit += op + "\n"
        else:
            print(op)
            raise ValueError(f"Unknown gate {gate} in the circuit")

    return stim.Circuit(new_circuit)

def _is_measurement_noisy(op: str) -> bool:
    gate = op.lstrip().split(" ")[0]
    assert 'M' in gate, f"{op=} is not a measurement"
    return True if '(' in gate else False

def get_erased_circuit_on_CNOTs(circuit: stim.Circuit, e: float, e1: float = 0, r=0) -> stim.Circuit:
    """
    e is erasure probability per operation
    WHAT DOES THIS DO??? is it different than get_erased_circuit?
    """
    
    erasures = np.zeros(circuit.num_qubits)
    circuit_list = str(circuit).split("\n")
    new_circuit = ""

    for i, op in enumerate(circuit_list):
        if NEW_CYCLE_STRING in op:
            r-=1
        if op.split(" ")[0] in SINGLE_QUBIT_GATES + TWO_QUBIT_CLIFFORD_GATES and r <= 0:
            for qubit in op.split(" "):
                if qubit.isdigit():
                    e_gate = e if op.split(" ")[0] in TWO_QUBIT_CLIFFORD_GATES else e1
                    if np.random.rand() < e_gate:
                        erasures[int(qubit)] = 1
                        # print(f"erased at {op=}")
            
            new_circuit += op + "\n"

            if op.split(" ")[0] in SINGLE_QUBIT_GATES:
                for qubit, erased in enumerate(erasures):
                    if erased > 0:
                        new_circuit += f"DEPOLARIZE1(0.74999) {int(qubit)}\n"
            elif op.split(" ")[0] in TWO_QUBIT_CLIFFORD_GATES:
                for qubit, erased in enumerate(erasures):
                    if erased > 0:
                        qubits = op.split()[1:]
                        interacting_qubit = [qubits[j+1] if j % 2 == 0 else qubits[j-1] for j in range(len(qubits)) if int(qubits[j]) == qubit][0]
                        new_circuit += f"DEPOLARIZE2(0.93749) {int(qubit)} {int(interacting_qubit)}\n"

            # reset erasures
            erasures = np.zeros(circuit.num_qubits)
        else:
            new_circuit += op + "\n"

    return stim.Circuit(new_circuit)

def erasure_collector(circuit: stim.Circuit, num_circuits, runs_per_circuit, json_metadata = {}) -> List[sinter.Task]:
    """
    run the circuit multiple times and collect statistics
    """
    task_list = []
    for _ in range(num_circuits):
        task_list.append(sinter.Task(
                    circuit = circuit,
                    json_metadata=json_metadata,
                    ))
        
    return task_list

def erasure_circuit_generator(circuit: stim.Circuit, e, r, num_circuits,
                               runs_per_circuit, json_metadata: dict = {}, e1=0, e_SPAM = None ,post_mask=None) -> Iterator[sinter.Task]:
    """
    run the circuit multiple times and collect statistics
    """
    for i in range(num_circuits):
        erased_circuit = get_erased_circuit(circuit, e=e, r=r, e1=e1, e_SPAM=e_SPAM)
        json_metadata["id"] = i
        yield sinter.Task(
            circuit=erased_circuit,
            postselection_mask=post_mask,
            json_metadata=json_metadata.copy(),  # there are no mutable values in the metadata
            
        )

def get_small_patch_qubits(circuit: stim.Circuit) -> List[str]:
    circuit_list = str(circuit).split("\n")
    R_list = []
    RX_list = []
    
    for op in circuit_list:
        gate = op.lstrip().split(" ")[0].split("(")[0]
        
        if gate == 'R' and not R_list:
            R_list = op.split()[1:]
        
        elif gate == 'RX' and not RX_list:
            RX_list = op.split()[1:]
        
        if R_list and RX_list:
            break

    return R_list + RX_list

def get_erasure_post_rate(circuit: stim.Circuit, e, r, erasure_post_qubits: List[str], e1=0, e_SPAM = None) -> float:
    """
    get the post selection success rate for a given circuit

    :erasure_post_qubits: a list of all qubits that are post-selected in the first r rounds. NOTICE THEY SHOULD BE STRINGS
    """
    circuit_list = str(circuit).split("\n")
    single_gate_num = 0
    two_gate_num = 0
    spam_gate_num = 0

    if e_SPAM is None:
        e_SPAM = e1

    for i, op in enumerate(circuit_list):
        gate = op.lstrip().split(" ")[0].split("(")[0]
        if gate in ['DEPOLARIZE1']:
            # count digits in op:
            single_gate_num += len([qubit for qubit in op.split(" ") if qubit.isdigit() and qubit in erasure_post_qubits])
        elif gate in TWO_QUBIT_ERRORS:
            two_gate_num += len([qubit for qubit in op.split(" ") if qubit.isdigit() and qubit in erasure_post_qubits])
        elif gate in SINGLE_QUBIT_ERRORS + MEASUREMENTS: # SPAM errors
            spam_gate_num += len([qubit for qubit in op.split(" ") if qubit.isdigit() and qubit in erasure_post_qubits])
        if NEW_CYCLE_STRING in op:
            # print("new cycle")
            r-=1
            if r == 0:
                break
    # print(single_gate_num, two_gate_num)
    return (1 - e) ** two_gate_num * (1 - e1) ** single_gate_num * (1 - e_SPAM) ** spam_gate_num

def remove_errors_from_injection(circuit: stim.Circuit, injection_rounds: int = 2, noisy_rounds = np.inf) -> stim.Circuit:
    """
    Remove all error gates from the first :injection_rounds: of the circuit, and after :noisy_rounds: more rounds
    """
    circuit_list = str(circuit).split('\n')
    clean_circuit = ""
    injection = injection_rounds
    for i, op in enumerate(circuit_list):
        if NEW_CYCLE_STRING in op:
            injection -= 1
        remove_errors = injection > 0 or injection <= -noisy_rounds
        if op.strip()[0] == 'M' and '(' in op and remove_errors:
            clean_circuit += op.split('(')[0] + op.split(')')[1] + '\n'
            continue
        if 'DEPOLARIZE' not in op and 'ERROR' not in op or not remove_errors:
            clean_circuit += op + '\n'
    return stim.Circuit(clean_circuit)

def immunify_qubits(circuit, immune_qubits, p_e = 1e-4):
    """
    go over the circuit and change the error rate of the qubits in immune_qubits to p_e
    for single_qubit_errors or measurement gates, change only the immune_qubits, 
    and leave the others with the same error rate
    for DEPOLARIZE2, change only if both qubits are in immune_qubits
    """
    new_circuit = stim.Circuit()
    for operation in circuit:
        noisy_operation = (
            stim.GateData(operation.name).is_noisy_gate 
            and (operation.num_measurements==0 or len(operation.gate_args_copy()) > 0)
            )
        if not noisy_operation:
            new_circuit.append(operation)
            continue
        for target_group in operation.target_groups():
            if all([int(qubit.qubit_value) in immune_qubits for qubit in target_group]):
                # print('immune qubit')
                new_op = stim.CircuitInstruction(operation.name, target_group, gate_args=[p_e])
                new_circuit.append(new_op)
            else:
                # print('non-immune qubit')
                new_op = stim.CircuitInstruction(operation.name, target_group, gate_args=operation.gate_args_copy())
                new_circuit.append(new_op)
    return new_circuit