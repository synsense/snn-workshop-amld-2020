from typing import Union
from pathlib import Path
from pprint import pprint
import os
import numpy as np
import pandas as pd
from rockpool import TSContinuous
from ECG import recordings

DT = 0.002_778

# - Input signal properties
# These recordings are not taken at MLII and V1 leads:
omit_recordings = [100, 102, 103, 104, 114, 117, 123, 124]
use_targets = [0, 1, 2, 3, 4, 5]
params_signal = {
    # "load_path": signal_path,
    "p_normal": 0.75,  # Probability of normal input rhythm (ignore if not uniform class probs)
    "include": {"target": use_targets},  # - Which target classes to use
    "exclude": {"is_used": True, "recording": omit_recordings},
    "min_len_segment": 5,  # Minimum number of beats in continuous anomaly segment
    "max_len_segment": 10,  # Maximum number of beats in continuous anomaly segment
    "continuous_segments": True,  # If `True`, segments consist of contiguous heart beats
    "min_anomal_per_seg": 3,
    "batchsize": 500,  # Number trials per test or validation batch
    "same_tgt_segments": False,  # Every beat in a segment is of same class
    "same_rec_segments": False,  # Segments are also of same recording (not only same class) - only relevant if use_cont_segments is False
}


class ECGDataLoader:
    def __init__(self):
        self.ecg_recordings = recordings.ECGRecordings()
        self.params = params_signal

        # - dict that maps classes from recordings to those used in plot
        try:
            target_classes = set(params_signal.pop("target_classes"))
        except KeyError:
            # - Infer target classes from probabilities
            target_classes = set(params_signal["target_probs"].keys())

        self.remap_targets = {k: v for v, k in enumerate(sorted(target_classes))}
        self.target_idcs = {
            tgt: idx - 1 for tgt, idx in self.remap_targets.items() if idx != 0
        }
        self.target_names = {
            tgt: self.ecg_recordings.target_names[tgt] for tgt in self.remap_targets
        }

    def get_batch_generator(self, num_beats: int, batchsize: int):

        # - Get data
        annotations, signal = self.ecg_recordings.provide_data(num_beats, **self.params)
        num_batches = int(np.ceil(annotations.index.size / batchsize))
        # - Make sure to not divide segments
        segment_ids = np.unique(annotations.segment_id)
        num_segs_batch = int(np.ceil(segment_ids.size / num_batches))
        idcs_seg_split = np.arange(1, num_batches) * num_segs_batch
        idcs_seg_split = idcs_seg_split[idcs_seg_split < segment_ids.size]
        # First segments of batches
        first_seg_ids = segment_ids[idcs_seg_split]
        # Indices where annotations are split into batches
        idcs_split = [
            np.where(annotations.segment_id == id_seg)[0][0] for id_seg in first_seg_ids
        ]
        # Will iterate over split annotations
        iterator = np.split(annotations, idcs_split)

        timestep_start = 0
        is_first = True
        for i_batch, it in enumerate(iterator):

            print(f"\n\tBatch {i_batch + 1} of {num_batches}")

            ann_batch = it
            signal_batch = signal[
                ann_batch.idx_start_new.iloc[0] : ann_batch.idx_end_new.iloc[-1]
            ]

            batch = self._create_batch(
                signal=signal_batch,
                annotations=ann_batch,
                i_batch=i_batch,
                timestep_start=timestep_start,
                is_first=is_first,
                is_last=i_batch == num_batches - 1,
                min_anomal_per_seg=self.params_signal["min_anomal_per_seg"],
            )
            timestep_start += batch.num_timesteps
            is_first = False
            yield batch

    def _get_target(self, ann_batch):
        return self.ecg_recordings.generate_target(
            ann_batch.index, map_target=self.remap_targets, boolean_raster=True
        )

    def _create_batch(
        self,
        signal,
        annotations,
        i_batch,
        timestep_start,
        is_first,
        is_last,
        min_anomal_per_seg,
    ):
        target_batch = self._get_target(annotations)

        return ECGBatch(
            n_id=i_batch,
            annotations=annotations,
            inp_data=signal,
            tgt_data=target_batch,
            timestep_start=timestep_start,
            is_first=is_first,
            is_last=is_last,
            dt=DT,
            min_anomal_per_seg=min_anomal_per_seg,
            target_idcs=self.target_idcs,
            target_names=self.target_names,
        )

    @property
    def dt(self):
        return DT


class ECGBatch:
    def __init__(
        self,
        n_id,
        annotations,
        inp_data,
        tgt_data,
        timestep_start,
        is_first,
        is_last,
        dt,
        min_anomal_per_seg,
        target_idcs,
        target_names,
    ):
        self.str_id = "{:03d}".format(n_id)
        self.annotations = annotations
        self.dt = dt
        self.is_first = is_first
        self.is_last = is_last
        self.min_anomal_per_seg = min_anomal_per_seg
        self.target_idcs = target_idcs
        self.target_names = target_names
        self._generate_timeseries(inp_data, tgt_data, timestep_start)

    def _generate_timeseries(self, inp_data, tgt_data, timestep_start):
        num_timesteps = inp_data.shape[0]
        self.times = (np.arange(num_timesteps) + timestep_start) * self.dt
        t_stop = (timestep_start + num_timesteps) * self.dt
        self.input = TSContinuous(self.times, inp_data, t_stop=t_stop)
        self.target = TSContinuous(self.times, tgt_data)
        self.num_timesteps = self.times.size
        self.duration = self.num_timesteps * self.dt
