import random
import time

import sinter
import stim
from latte.vec_sim import VecSim


class VecInterceptSampler(sinter.Sampler):
    """Samples while overriding S rotations with powers of T.

    This sampler is highly specialized for injection circuits where
    consistent powers of T all distill correctly.

    Uses a vector simulator to make it possible to perform non
    stabilizer gates.
    """

    def __init__(self, logical_x : list[int], logical_z : list[int]):
        self.logical_x = logical_x
        self.logical_z = logical_z

    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        return CompiledVecInterceptSampler(task, self.logical_x, self.logical_z)


class CompiledVecInterceptSampler(sinter.CompiledSampler):
    def __init__(self, task: sinter.Task, logical_x : list[int], logical_z : list[int]):
        self.task = task
        self.logical_x = logical_x
        self.logical_z = logical_z

    def sample(self, shots: int, measured_operator = 'H') -> sinter.AnonTaskStats:
        result = sinter.AnonTaskStats()
        for _ in range(shots):
            result += sample_circuit_with_vec_sim(
                self.task.circuit,
                self.logical_x,
                self.logical_z,
                measured_operator,
            )
        return result


def sample_circuit_with_vec_sim(circuit: stim.Circuit, 
                                logical_x : list[int], 
                                logical_z : list[int],
                                measured_operator = 'H') -> sinter.AnonTaskStats:
    t0 = time.monotonic()
    sim = VecSim()
    measurements = []
    detectors = []
    observables = []
    sweep_bits = {
        b: False
        for b in range(circuit.num_sweep_bits)
    }

    discard_shot = False
    for q in range(circuit.num_qubits):
        sim.do_qalloc_z(q)
    for line_count, inst in enumerate(circuit):
        if inst.name == 'MPP':
            for terms in inst.target_groups():
                combined_targets = []
                for term in terms:
                    if term.is_y_target:
                        combined_targets.append(stim.target_pauli(term.value, 'Z'))
                    else:
                        combined_targets.append(term)
                    combined_targets.append(stim.target_combiner())
                combined_targets.pop()
                if all(term.is_y_target for term in terms):
                    if measured_operator == 'H':
                        sim.do_s_obs({(q):"Z" for q in logical_z}, sign=1)
                        sim.do_t_obs({(q):"X" for q in logical_x}, sign=1)
                    elif measured_operator == 'HXY':
                        sim.do_t_obs({(q):"Z" for q in logical_z}, sign=1)
                        sim.do_s_obs({(q):"X" for q in logical_x}, sign=1)
                    else:
                        raise Exception(f"Unknown measured operator: {measured_operator}")
                sim.do_stim_instruction(
                    stim.CircuitInstruction('MPP', combined_targets, inst.gate_args_copy()),
                    sweep_bits=sweep_bits,
                    out_measurements=measurements,
                    out_detectors=detectors,
                    out_observables=observables,
                )
                if all(term.is_y_target for term in terms):
                    if measured_operator == 'H':
                        sim.do_t_obs({(q):"X" for q in logical_x},sign=-1)
                        sim.do_s_obs({(q):"Z" for q in logical_z},sign=-1)
                    elif measured_operator == 'HXY':
                        sim.do_t_obs({(q):"X" for q in logical_x},sign=-1)
                        sim.do_s_obs({(q):"Z" for q in logical_z},sign=-1)

        elif inst.name == 'DETECTOR':
            b = False
            for q in inst.targets_copy():
                assert q.is_measurement_record_target
                b ^= measurements[q.value]
            if b:
                print(f"{inst=} {q.value=} {line_count=}")
                discard_shot = True
                break
        else:
            sim.do_stim_instruction(
                inst,
                sweep_bits=sweep_bits,
                out_measurements=measurements,
                out_detectors=detectors,
                out_observables=observables,
            )
    t1 = time.monotonic()
    # measurement_results = [int(i) for i in measurements]
    # int(measurement_results)
    if discard_shot:
        return sinter.AnonTaskStats(discards=1, shots=1, seconds=t1 - t0)
    if any(observables):
        return sinter.AnonTaskStats(errors=1, shots=1, seconds=t1 - t0)
    return sinter.AnonTaskStats(shots=1, seconds=t1 - t0)
