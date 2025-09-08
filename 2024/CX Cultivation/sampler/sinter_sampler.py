import time
import numpy as np
import stim
import sinter
import pymatching
import math
import collections


class PostSelectionSampler(sinter.Sampler):
    """Predicts obs aren't flipped. Discards shots with any detection events."""
    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        return CompiledPostSelectionSampler(task)


class CompiledPostSelectionSampler(sinter.CompiledSampler):
    def __init__(self, task: sinter.Task):
        self.stim_sampler = task.circuit.compile_detector_sampler()
        self.dem = task.circuit.detector_error_model()
        self.post_selected_detectors = np.zeros(self.dem.num_detectors)

        detector_coords = self.dem.get_detector_coordinates()

        for idx, coord in (detector_coords.items()):
            if len(coord) == 0 or (coord[0]):
                self.post_selected_detectors[idx] = 1

        self.matching_graph = pymatching.Matching.from_detector_error_model(self.dem)

    def sample(self, max_shots: int) -> sinter.AnonTaskStats:
        t0 = time.monotonic()
        data, logical_obs = self.stim_sampler.sample(shots=max_shots, separate_observables=True)
        data_filter = np.all(data[:, self.post_selected_detectors == 1] == 0, axis=1)
        post_selected_data = data[data_filter, :]
        post_selected_logical_obs = logical_obs[data_filter]

        matching_data = self.matching_graph.decode_batch(post_selected_data)
        errors = np.sum(np.any(matching_data != post_selected_logical_obs, axis=1))
        discards = max_shots - post_selected_data.shape[0]

        t1 = time.monotonic()

        return sinter.AnonTaskStats(
            shots = max_shots,
            errors = int(errors),
            discards = int(discards),
            seconds = t1 - t0,
        )


class GapSampler(sinter.Sampler):
    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        return CompiledGapSampler(task, True)

class GapCompleteSampler(sinter.Sampler):
    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        return CompiledGapSampler(task, False)


class CompiledGapSampler(sinter.CompiledSampler):
    def __init__(self, task: sinter.Task, postselect : bool):
        self.postselect = postselect
        self.task = task
        circuit = task.circuit.copy()
        self._original_dem = task.circuit.detector_error_model().flattened()
        for idx in range((-self._original_dem.num_detectors)%8 + 2):
            circuit.append("DETECTOR", [],[0])        

        self.postselected_detectors = []
        detector_coords = self._original_dem.get_detector_coordinates()
        for idx, coord in (detector_coords.items()):
#             if len(coord) == 0 or (coord[0]):
            if len(coord) == 0 and self.postselect:    
                self.postselected_detectors.append(idx)
                
        
        self.gap_dem = _dem_with_obs_detector(self._original_dem)
        self.gap_circuit = circuit
        assert self.gap_circuit.num_detectors == self.gap_dem.num_detectors
        self.num_dets = self.gap_circuit.num_detectors
        self.num_det_bytes = -(-self.num_dets // 8)
        self._discard_mask = np.packbits(np.array([k in self.postselected_detectors for k in range(self.num_dets)], dtype=np.bool_), bitorder='little')
        self.gap_circuit_sampler = self.gap_circuit.compile_detector_sampler()
        self.gap_decoder = pymatching.Matching.from_detector_error_model(self.gap_dem)
        
        self._obs_det_byte_ZZ = 1
        self._obs_det_byte_XX = 2
        
        edge = next(iter(self.gap_decoder.to_networkx().edges.values()))
        try:
            edge_w = edge['weight']
            edge_p = edge['error_probability']
            self.decibels_per_w = -math.log10(edge_p / (1 - edge_p)) * 10 / edge_w
        except:
            self.decibels_per_w = 1


    def sample(self, shots: int) -> sinter.AnonTaskStats:
        t0 = time.monotonic()
        dets, actual_obs = self.gap_circuit_sampler.sample(shots, separate_observables=True, bit_packed=True)

        keep_mask = ~np.any(dets & self._discard_mask, axis=1)
        dets = dets[keep_mask]
        actual_obs = actual_obs[keep_mask]
        assert actual_obs.shape[1] == 1
        actual_obs = actual_obs[:, 0]
        predictions, gaps = self._decode_batch_overwrite_last_byte(bit_packed_dets=dets)
        errors = predictions ^ actual_obs
        counter = collections.Counter()
        for gap, err in zip(gaps, errors):
            counter[f'E{round(gap)}' if err else f'C{round(gap)}'] += 1
        t1 = time.monotonic()
        
        stats = sinter.AnonTaskStats(
            shots=shots,
            errors=np.count_nonzero(errors),
            discards=shots - np.count_nonzero(keep_mask),
            seconds=t1 - t0,
            custom_counts=counter,
        )
        return stats

    def _decode_batch_overwrite_last_byte(self, bit_packed_dets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bit_packed_dets[:, -1] |= self._obs_det_byte_ZZ
        bit_packed_dets[:, -1] |= self._obs_det_byte_XX
        _, ZZ1_XX1_weights = self.gap_decoder.decode_batch(bit_packed_dets, return_weights=True, bit_packed_shots=True, bit_packed_predictions=True)
        
        bit_packed_dets[:, -1] ^= self._obs_det_byte_ZZ
        _, ZZ0_XX1_weights = self.gap_decoder.decode_batch(bit_packed_dets, return_weights=True, bit_packed_shots=True, bit_packed_predictions=True)
        
        bit_packed_dets[:, -1] ^= self._obs_det_byte_XX
        _, ZZ0_XX0_weights = self.gap_decoder.decode_batch(bit_packed_dets, return_weights=True, bit_packed_shots=True, bit_packed_predictions=True)

        bit_packed_dets[:, -1] ^= self._obs_det_byte_ZZ
        _, ZZ1_XX0_weights = self.gap_decoder.decode_batch(bit_packed_dets, return_weights=True, bit_packed_shots=True, bit_packed_predictions=True)
        
        probabilities = np.column_stack((ZZ0_XX0_weights, ZZ1_XX0_weights,
                                         ZZ0_XX1_weights, ZZ1_XX1_weights))
        # Get the indices of the maximum probabilities for each position
        max_indices = np.argmin(probabilities, axis=1)
        max_probs = probabilities[np.arange(len(probabilities)), max_indices]

        # Create a copy of probabilities and set the maximum probabilities to negative infinity
        probabilities_copy = probabilities.copy()
        probabilities_copy[np.arange(len(probabilities)), max_indices] = np.inf

        # Get the indices of the second maximum probabilities for each position
        second_max_indices = np.argmin(probabilities_copy, axis=1)
        second_max_probs = probabilities_copy[np.arange(len(probabilities_copy)), second_max_indices]

        # Compute the differences between the maximum and second maximum probabilities
        gaps : np.ndarray = np.abs((second_max_probs - max_probs)* self.decibels_per_w)
        predictions: np.ndarray = max_indices
        return predictions, gaps

def _dem_with_obs_detector(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    new_dem = stim.DetectorErrorModel()
    for idx in range((-dem.num_detectors)%8):
        new_dem.append("detector",[0], [stim.target_relative_detector_id(dem.num_detectors + idx)])
    non_observable_detector_count = math.ceil(dem.num_detectors/8) * 8
                       
    obs_det_ZZ = stim.target_relative_detector_id(non_observable_detector_count)
    obs_det_XX = stim.target_relative_detector_id(non_observable_detector_count + 1)
    new_dem.append('detector', [0], [obs_det_ZZ])
    new_dem.append('detector', [0], [obs_det_XX])
    
    for inst in dem:
        if inst.type == 'error':
            targets = inst.targets_copy()
            new_targets = []
            for t in targets:
                if t.is_logical_observable_id():
                    if t.val == 0 :
                        new_targets.append(obs_det_ZZ)
                    elif t.val == 1:
                        new_targets.append(obs_det_XX)
                    else:
                        raise Exception(f"Unknown logical observable val {t.val}")
                new_targets.append(t)
            new_dem.append('error', inst.args_copy(), new_targets)
        else:
            new_dem.append(inst)
    return new_dem    

class SingleObservableGapSampler(sinter.Sampler):
    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        return CompiledSingleObservableGapSampler(task, False)

class CompiledSingleObservableGapSampler(sinter.CompiledSampler):
    def __init__(self, task: sinter.Task, postselect : bool):
        self.postselect = postselect
        self.task = task
        circuit = task.circuit.copy()
        self._original_dem = task.circuit.detector_error_model().flattened()
        for idx in range((-self._original_dem.num_detectors)%8 + 1):
            circuit.append("DETECTOR", [],[0])        

        self.postselected_detectors = []
        detector_coords = self._original_dem.get_detector_coordinates()
        for idx, coord in (detector_coords.items()):
            if len(coord) == 0 and self.postselect:    
                self.postselected_detectors.append(idx)
        
        self.gap_dem = _dem_with_obs_detector_single_obs(self._original_dem)
        self.gap_circuit = circuit
        assert self.gap_circuit.num_detectors == self.gap_dem.num_detectors
        self.num_dets = self.gap_circuit.num_detectors
        self.num_det_bytes = -(-self.num_dets // 8)
        self._discard_mask = np.packbits(np.array([k in self.postselected_detectors for k in range(self.num_dets)], dtype=np.bool_), bitorder='little')
        self.gap_circuit_sampler = self.gap_circuit.compile_detector_sampler()
        self.gap_decoder = pymatching.Matching.from_detector_error_model(self.gap_dem)
        
        self._obs_det_byte = 1        
        edge = next(iter(self.gap_decoder.to_networkx().edges.values()))
        try:
            edge_w = edge['weight']
            edge_p = edge['error_probability']
            self.decibels_per_w = -math.log10(edge_p / (1 - edge_p)) * 10 / edge_w
        except:
            self.decibels_per_w = 1


    def sample(self, shots: int) -> sinter.AnonTaskStats:
        t0 = time.monotonic()
        dets, actual_obs = self.gap_circuit_sampler.sample(shots, separate_observables=True, bit_packed=True)

        keep_mask = ~np.any(dets & self._discard_mask, axis=1)
        dets = dets[keep_mask]
        actual_obs = actual_obs[keep_mask]
        assert actual_obs.shape[1] == 1
        actual_obs = actual_obs[:, 0]
        predictions, gaps = self._decode_batch_overwrite_last_byte(bit_packed_dets=dets)
        errors = predictions ^ actual_obs
        counter = collections.Counter()
        for gap, err in zip(gaps, errors):
            counter[f'E{round(gap)}' if err else f'C{round(gap)}'] += 1
        t1 = time.monotonic()
        
        stats = sinter.AnonTaskStats(
            shots=shots,
            errors=np.count_nonzero(errors),
            discards=shots - np.count_nonzero(keep_mask),
            seconds=t1 - t0,
            custom_counts=counter,
        )
        return stats

    def _decode_batch_overwrite_last_byte(self, bit_packed_dets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bit_packed_dets[:, -1] |= self._obs_det_byte
        _, on_weights = self.gap_decoder.decode_batch(bit_packed_dets, return_weights=True, bit_packed_shots=True, bit_packed_predictions=True)
        
        bit_packed_dets[:, -1] ^= self._obs_det_byte
        _, off_weights = self.gap_decoder.decode_batch(bit_packed_dets, return_weights=True, bit_packed_shots=True, bit_packed_predictions=True)

        # Compute the differences between the maximum and second maximum probabilities
        gaps: np.ndarray = np.abs((on_weights - off_weights) * self.decibels_per_w)
        predictions: np.ndarray = on_weights < off_weights
        return predictions, gaps

def _dem_with_obs_detector_single_obs(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    new_dem = stim.DetectorErrorModel()
    for idx in range((-dem.num_detectors)%8):
        new_dem.append("detector",[0], [stim.target_relative_detector_id(dem.num_detectors + idx)])
    non_observable_detector_count = math.ceil(dem.num_detectors/8) * 8
                       
    obs_det = stim.target_relative_detector_id(non_observable_detector_count)
    new_dem.append('detector', [0], [obs_det])
    
    for inst in dem:
        if inst.type == 'error':
            targets = inst.targets_copy()
            new_targets = []
            for t in targets:
                if t.is_logical_observable_id():
                    if t.val == 0 :
                        new_targets.append(obs_det)
                    else:
                        raise Exception(f"Unknown logical observable val {t.val}")
                new_targets.append(t)
            new_dem.append('error', inst.args_copy(), new_targets)
        else:
            new_dem.append(inst)
    return new_dem    

def sinter_samplers() -> dict[str, sinter.Sampler]:
    return {"PostSelectionSampler" : PostSelectionSampler(),
            "GapSampler" : GapSampler(),
            "GapCompleteSampler" : GapCompleteSampler(),
            "SingleObservableGapSampler" : SingleObservableGapSampler(),
           }