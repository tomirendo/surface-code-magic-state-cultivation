import stim
import sys
# Pauli Gates
PAULI_GATES = ["I", "X", "Y", "Z"]

# Single Qubit Clifford Gates
SINGLE_QUBIT_CLIFFORD_GATES = [
    "C_XYZ", "C_ZYX", "H", "H_XY", "H_XZ", "H_YZ", 
    "S", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", 
    "SQRT_Z", "SQRT_Z_DAG", "S_DAG"
]

SINGLE_QUBIT_DICT = {
    "I": "$I$",
    "X": "$X$",
    "Y": "$Y$",
    "Z": "$Z$",
    "H": "$H$",
    "S": "$S$",
    "S_DAG": r"$S^\dagger$",
    "SQRT_X": r"$\sqrt{X}$",
    "SQRT_X_DAG": r"$\sqrt{X}^\dagger$",
    "SQRT_Y": r"$\sqrt{Y}$",
    "SQRT_Y_DAG": r"$\sqrt{Y}^\dagger$",
    "SQRT_Z": r"$\sqrt{Z}$",
    "SQRT_Z_DAG": r"$\sqrt{Z}^\dagger$",
    "C_XYZ": r"$C_{XYZ}$",
    "C_ZYX": r"$C_{ZYX}$",
    "H_XY": r"$H_{XY}$",
    "H_XZ": r"$H_{XZ}$",
    "H_YZ": r"$H_{YZ}$"
}

SINGLE_QUBIT_GATES = PAULI_GATES + SINGLE_QUBIT_CLIFFORD_GATES

COLLAPSING_GATES = [
    "M", "MR", "MRX", "MRY", "MRZ", 
    "MX", "MY", "MZ", "R", "RX", "RY", "RZ"
]

MEASUREMENTS = [
    "M", "MR", "MRX", "MRY", "MRZ", 
    "MX", "MY", "MZ",
]

MEASUREMENT_DICT = {
    "M": "$M_Z$",
    "MR": "$M_Z$",
    "MRX": "$M_X$",
    "MRY": "$M_Y$",
    "MRZ": "$M_Z$",
    "MX": "$M_X$",
    "MY": "$M_Y$",
    "MZ": "$M_Z$",
}
  

RESETS = [
    "R", "RX", "RY", "RZ", "MR", "MRX", "MRY", "MRZ"
]

RESET_DICT = {
    "R": "$R$",
    "RX": "$R_X$",
    "RY": "$R_Y$",
    "RZ": "$R_Z$",
    "MR": "$R$",
    "MRX": "$R_X$",
    "MRY": "$R_Y$",
    "MRZ": "$R_Z$"
}

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


def convert_stim_to_qpic(stim_circuit):
    qpic_string = ""

    qubits = list(range(stim_circuit.num_qubits))

    # circuit_list = str(stim_circuit).split('\n')
    for op in stim_circuit:
        gate = op.name
        target_qubits = op.target_groups()
        if gate == "TICK":
            qpic_string += f"{' '.join(map(str, qubits))} TOUCH\n"
        if gate == "QUBIT_COORDS":
            qpic_string += f"{(target_qubits[0][0]).qubit_value} W owire\n"
        if gate in SINGLE_QUBIT_DICT.keys():
            for qubit in target_qubits:
                qpic_string += f"{qubit[0].qubit_value} G {{\\scriptsize {SINGLE_QUBIT_DICT[gate]}}}\n"
        if gate in TWO_QUBIT_CLIFFORD_GATES:
            if gate == "CNOT" or "CX":
                for qubit_pair in target_qubits:
                    qpic_string += f"{qubit_pair[0].qubit_value} +{qubit_pair[1].qubit_value}\n"
            else:
                raise ValueError(f"2Q gate {gate} not supported in qpic2stim yet")
        if gate in MEASUREMENTS:
            for qubit in target_qubits:
                q = qubit[0].qubit_value
                qpic_string += f"{q} M {{\\scriptsize {MEASUREMENT_DICT[gate]}}} {q}:owire\n"
        if gate in RESETS:
            for qubit in target_qubits:
                q = qubit[0].qubit_value
                qpic_string += f"{q} G {{\\scriptsize {RESET_DICT[gate]}}} {q}:qwire\n"
    return qpic_string


if __name__ == "__main__":
    stim_circuit = stim.Circuit.from_file(sys.argv[1])
    qpic_string = convert_stim_to_qpic(stim_circuit)
    print(qpic_string)