# uses data from https://physionet.org/physiobank/database/mitdb/

from typing import Optional, Union, List, Iterable, Dict, Any, Set
from pathlib import Path
import os
import random
from warnings import warn

import numpy as np
import scipy.signal as ssg
import pandas as pd
from matplotlib import pyplot as plt
import wfdb

from rockpool.utilities import ArrayLike

ecg_dir = Path(__file__).parent
load_path = ecg_dir / "ecg_recordings" / "physiobank" / "mitdb"
save_path = ecg_dir / "ecg_recordings"
save_path_hpf = ecg_dir / "ecg_recordings" / "hpf"
save_path_no_afib = ecg_dir / "ecg_recordings" / "no_afib"
save_path_no_afib_hpf = ecg_dir / "ecg_recordings" / "no_afib" / "hpf"

DT = 1.0 / 360.0
DEF_HP_FREQ = 0.1

# #list of databases from online source
# wfdb.get_dbs()

translate_annotations = {
    "N": 0,  # Normal beat
    "L": 1,  # Left bundle branch block beat
    "R": 2,  # Right bundle branch block beat
    "V": 3,  # Premature ventricular contraction
    "/": 4,  # Paced beat
    "A": 5,  # Atrial premature beat
    "f": 6,  # Fusion of paced and normal beat
    "F": 7,  # Fusion of ventricular and normal beat
    "!": 8,  # Ventricular flutter wave (*) - Treat as abnormal "beat"
    "a": 9,  # Aberrated atrial premature beat
    "j": 10,  # Nodal (junctional) escape beat
    "E": 11,  # Ventricular escape beat
    "J": 12,  # Nodal (junctional) premature beat
    "e": 13,  # Atrial escape beat
    "S": 14,  # Supraventricular premature or ectopic beat (atrial or nodal)
    "Q": 15,  # Unclassifiable beat
    ## -- Added annotaitions
    "|": 16,  # Isolated QRS-like artifact * - Search beats that have this and mark as anomaly
    # "||": 16,  # Beat containing "|" annotation
    "x": 17,  # Non-conducted P-wave (blocked APC) * - Search beats that have this and mark as anomaly
    # "xx": 17,  # Beat containing "x" annotation
    "(AFIB": 18,  # Beat marked as AFIB in aux_notes
    "(AFL": 19,  # Beat marked as AFL in aux_nots
    "(PREX": 20,  # Beat marked as PREX in aux_notes
    ## -- Not actual beats:
    "~": 21,  # Change in signal quality * - identify which one is a start and which one the end of a bad signal, mark affected beats with boolean
    "+": 22,  # Rhythm change * - Check ann_wfdb.aux_note for details, mark groups of beats for some of the notes
    '"': 23,  # Comment annotation * - ignore for now
}

target_names = {
    0: "Normal beat",
    1: "Left bundle branch block beat",
    2: "Right bundle branch block beat",
    3: "Premature ventricular contraction",
    4: "Paced beat",
    5: "Atrial premature beat",
    6: "Fusion of paced and normal beat",
    7: "Fusion of ventricular and normal beat",
    8: "Ventricular flutter wave",
    9: "Aberrated atrial premature beat",
    10: "Nodal (junctional) escape beat",
    11: "Ventricular escape beat",
    12: "Nodal (junctional) premature beat",
    13: "Atrial escape beat",
    14: "Supraventricular premature or ectopic beat (atrial or nodal)",
    15: "Unclassifiable beat",
    ## -- Added annotaitions
    16: "Isolated QRS-like artifact",
    # "||": 16,  # Beat containing "|" annotation
    17: "Non-conducted P-wave (blocked APC)",
    # "xx": 17,  # Beat containing "x" annotation
    18: "Beat marked as AFIB in aux_notes",
    19: "Beat marked as AFL in aux_nots",
    20: "Beat marked as PREX in aux_notes",
    ## -- Not actual beats:
    21: "Change in signal quality",
    22: "Rhythm change",
    23: "Comment annotation",
}

beat_annotations = [k for k, v in translate_annotations.items() if v <= 20]
plus_annotations = ["(AFIB", "(AFL", "(PREX"]
# plus_annotations = ["(AFL", "(PREX"]
# plus_annotations = []
inner_annotations = ["|", "x"]  # Non-beat annotations that are copied to full beat
# inner_annotations = []  # Non-beat annotations that are copied to full beat

# -- "[]"s: (only recording 207) Beats inside are marked with "!", so ignore
## -- "+"s: Look at ann_wfdb.aux_note:
# "(AFIB" - Atrial fibrillation: Beats do not seem to be marked - Train output to detect?
# "(AFL" - Atrial flutter: Beats do not seem to be marked - Train output to detect?
# "(PREX" - Pre-excitation (Wolff-Parkinson-White, WPW), usually not dangerous. Anyway, mark those beats and use 1 output to detect
# "(BII" - 2nd degree heart block, not marked in beats, only occurs in recording 231, maybe ignore for now
# "(N" - Normal sinus rhythms begin - ignore
# "(P" - Paced beats begin - marked with "/" anyway, so ignore
# "(B", "(T" - Ventricular bi-/trigeminy: Every second/third beat is marked as "V" anyway, so ignore
# "(VT" - Ventricular tachycardia: Beats seem to be marked with "V" (or "F") anyway, so ignore
# "(SVTA" - Supraventricular tachyarrhythmia: Beats are marked with "A", "a", J" or "V", so ignore
# "(NOD" - Nodal (A-V junctional) rhythm: Beats seem to be marked with "J" or "j", so ignore
# "(IVR" - Idioventricular rhythm: Beats seem to be marked with "V" or "E", so ignore
# "(VFL" - Ventricular flutter: Beats seem to be marked with "!", ignore


def download_db(name_db: str = "afdb"):
    # - list of recordings of a database
    wfdb.get_record_list("afdb")
    # Download database
    wfdb.dl_database("afdb", "physiobank/afdb/", "all", "all")


### --- Functions for extracting downloaded data and processing annotations


def _load_from_file(
    id_rec: Union[int, str], load_path: Union[str, Path]
) -> (np.ndarray, wfdb.Annotation, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    _load_from_file - Load data and annotations for specific recording from file.
                      Determine start indices for beats and assign preliminary targets
                      based on annotations.
    :param id_rec:  ID of recording that is to be loaded
    :param load_path:  Directory containig the recording
    :return:
        2D-float-array ([timesteps x channels]) with recorded ecg signal
        Annotation object for loaded recording
        1D-str-array of all (main) annotations as strings
        1D-int-array of indices in data corresponding to annotations
        1D-int-array of targets corresponding to heart beats
        1D-int-array of indices in data corresponding to beginnings of heart beats
    """
    # - Recorded data
    ecg_data = wfdb.io.rdrecord(os.path.join(load_path, str(id_rec))).p_signal
    # - Annodation data
    ann_wfdb = wfdb.io.rdann(os.path.join(load_path, str(id_rec)), "atr")
    all_annots = np.array(ann_wfdb.symbol)
    # - Indices in data corresponding to annotations
    idcs_all_annots = ann_wfdb.sample
    # - Annotations referring to individual heart beats (as opposed to remark inside beat)
    is_beat_annot = np.isin(all_annots, beat_annotations)
    # - Translate annotations to corresponding IDs
    beat_targets = np.fromiter(
        (translate_annotations[annot] for annot in all_annots[is_beat_annot]), int
    )
    # A beat is considered to start with a beat annotation - get corresponding indices
    idcs_beat_starts = idcs_all_annots[is_beat_annot]

    return (
        ecg_data,
        ann_wfdb,
        all_annots,
        idcs_all_annots,
        beat_targets,
        idcs_beat_starts,
    )


def _inner_annot_as_class(
    annot: str,
    all_annots: np.ndarray,
    idcs_all_annots: np.ndarray,
    idcs_beat_starts: np.ndarray,
    beat_targets: np.ndarray,
):
    """
    _inner_annot_as_class - Find occurances of a specific non-beat annotation and adapt
                            class of corresponding beat (in `beat_targets`) accordingly
                            if this beat is is normal.
    :param annot:  The non-beat annotation to be found
    :param all_annots:  str-array of all annotations for data
    :param idcs_all_annots:  int-array of indices corresponding to data
    :param idcs_beat_starts:  int-array of indices where beats start
    :param beat_targets:  int-rray of class IDs for beats
    """
    # Find beats corresponding to the annotation
    idcs_inner_annots = idcs_all_annots[np.where(all_annots == annot)[0]]
    # Indices in idcs_beat_starts of beat annotation preceding inner annotation
    idcs_beats = np.searchsorted(idcs_beat_starts, idcs_inner_annots) - 1
    # Overwrite beat label if normal beat
    do_overwrite = beat_targets[idcs_beats] == translate_annotations["N"]
    beat_targets[idcs_beats[do_overwrite]] = translate_annotations[annot]


def _process_aux_annotations(
    ann_wfdb: wfdb.Annotation,
    all_annots: np.ndarray,
    idcs_all_annots: np.ndarray,
    idcs_beat_starts: np.ndarray,
    beat_targets: np.ndarray,
    num_timesteps: int,
):
    """
    _process_aux_annotations - Include auxiliary annotations (marked with '+') and
                               update `beat_targets` if appropriate.
    :param ann_wfdb:  Annotations loaded from file.
    :param all_annots:  str-array of all annotations for data
    :param idcs_all_annots:  int-array of indices corresponding to data
    :param idcs_beat_starts:  int-array of indices where beats start
    :param beat_targets:  int-rray of class IDs for beats
    :param num_timesteps:  Number of timesteps (signal samples) for current recording
    """
    # Indices wrt idcs_all_annots where label is "+"
    idcs_plus = np.where(all_annots == "+")[0]
    # Auxiliary notes with more details about the "+" annotation - array to correct string encoding
    aux_notes = np.asarray(ann_wfdb.aux_note)
    # Indices wrt idcs_plus corresponding to "+" where relabeling is required
    idcs_relabel = np.where(np.isin(aux_notes[idcs_plus], plus_annotations))[0]
    # Indices wrt data where relabeling is required - start and end of "+"s
    idcs_relabel_start = idcs_all_annots[idcs_plus[idcs_relabel]]
    # Append number of data samples to idcs_all_annots to handle labelling until last beat
    idcs_all_annots_extd = np.r_[idcs_all_annots, num_timesteps]
    idcs_plus_extd = np.r_[idcs_plus, idcs_all_annots.size]
    idcs_relabel_end = idcs_all_annots_extd[idcs_plus_extd[idcs_relabel + 1]]
    # Beats where periods begin and end that are to be relabeled
    idcs_beat_plus_start = np.searchsorted(idcs_beat_starts, idcs_relabel_start) - 1
    idcs_beat_plus_end = np.searchsorted(idcs_beat_starts, idcs_relabel_end)
    # - Relabel beats
    for i_start, i_end, aux_annot in zip(
        idcs_beat_plus_start, idcs_beat_plus_end, aux_notes[idcs_plus[idcs_relabel]]
    ):
        beat_targets[i_start:i_end] = translate_annotations[aux_annot]


def _mark_bad_signal(
    ann_wfdb: wfdb.Annotation,
    all_annots: np.ndarray,
    idcs_all_annots: np.ndarray,
    idcs_beat_starts: np.ndarray,
) -> np.ndarray:
    """
    _mark_bad_signal - Return boolean array indicating which beats are flagged
                       as having bad signal quality.
    :param ann_wfdb:  Annotations loaded from file.
    :param all_annots:  str-array of all annotations for data
    :param idcs_all_annots:  int-array of indices corresponding to data
    :param idcs_beat_starts:  int-array of indices where beats start
    :return:
        1D-bool-array indicating which beats contain bad signal.
    """
    is_bad_beat = np.zeros(idcs_beat_starts.size, bool)
    idcs_quality_chg = np.where(all_annots == "~")[0]
    # - Parts where signal quality is marked as bad
    has_bad_qlty = ann_wfdb.subtype[idcs_quality_chg] > 0
    # - Points where signal quality turns from good to bad (i.e. has_bad_qlty is True and has been False before)
    has_worse_quality = has_bad_qlty & (np.r_[True, ~(has_bad_qlty[:-1])])
    idcs_bad_signal_start = idcs_all_annots[idcs_quality_chg[has_worse_quality]]
    # - Points where signal quality turns from bad to good
    idcs_bad_signal_end = idcs_all_annots[idcs_quality_chg[has_bad_qlty == False]]

    # - Determine beats with bad signal
    idcs_bad_beat_start = np.searchsorted(idcs_beat_starts, idcs_bad_signal_start) - 1
    idcs_bad_beat_end = np.searchsorted(idcs_beat_starts, idcs_bad_signal_end)
    for i_bad_start, i_bad_end in zip(idcs_bad_beat_start, idcs_bad_beat_end):
        is_bad_beat[i_bad_start:i_bad_end] = True

    return is_bad_beat


def _load_recording(id_rec: int, load_path: Union[str, Path]):
    """
    _load_recording - Load data and annotations for specific recording.
                      Generate target labels for beats from annotations and
                      generate array indicating which beats have bad signal.
    :param id_rec:  ID of recording that is to be loaded
    :param load_path:  Directory containig the recording
    :return:
        2D-float-array ([timesteps x channels]) with recorded ecg signal
        1D-int-array of targets corresponding to heart beats
        1D-int-array of indices in data corresponding to beginnings of heart beats
        1D-bool-array indicating which beats contain bad signal.
    """
    print("Processing recording #{}.".format(id_rec), end="\r")
    # - Load data for recording from file
    ecg_data, ann_wfdb, all_annots, idcs_all_annots, beat_targets, idcs_beat_starts = _load_from_file(
        id_rec, load_path
    )

    # - Replace annotation of normal beats with specific "non-beat annotations" if present
    for annot in inner_annotations:
        _inner_annot_as_class(
            annot, all_annots, idcs_all_annots, idcs_beat_starts, beat_targets
        )

    # - Find "+"s and change beat annotations of corresponding periods accordingly
    _process_aux_annotations(
        ann_wfdb=ann_wfdb,
        all_annots=all_annots,
        idcs_all_annots=idcs_all_annots,
        idcs_beat_starts=idcs_beat_starts,
        beat_targets=beat_targets,
        num_timesteps=ecg_data.shape[0],
    )

    # - Determine which beats are flagged as having bad signal quality
    is_bad_beat = _mark_bad_signal(
        ann_wfdb, all_annots, idcs_all_annots, idcs_beat_starts
    )

    return ecg_data, beat_targets, idcs_beat_starts, is_bad_beat


def extract_data_from_files(
    load_path: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None,
    freq_hp: Optional[float] = None,
) -> (pd.DataFrame, np.ndarray):
    """
    extract_data_from_files - Load ecg recordings with annotations. Generate
                              2D-array with ecg signal from all recordings and
                              DataFrame with information to all heartbeats.
    :param load_path:  Path to directory where data is stored.
    :param save_path:  If not `None`, save extracted data here.
    :param freq_hp:    If not `None`, apply a high-pass filter (2nd order Butterworth)
                       to each recording, with `freq_hp` as frequency
    :return:
        DataFrame with annotations for each beat (each beat one row)
        2D-array with ecg signal from all recordings ([# time steps x # ecg channels])
    """
    # - Available recordings
    lFiles = os.listdir(load_path)
    vRecordingIDs = np.unique([strFilename.split(".")[0] for strFilename in lFiles])
    # - Lists for collecting recordings, Annotations, corresponding indices
    signal_list: List[np.ndarray] = []
    targets_coll: List[int] = []
    idcs_beat_starts_coll = []
    rec_ids_coll = []
    is_bad_beat_coll = []

    # - Prepare high-pass filter
    if freq_hp is not None:
        b_hp, a_hp = ssg.butter(2, 2.0 * DT * freq_hp, btype="highpass")

    # - Iterate over recordings. Add info to DataFrame and collect signal in array
    idx_start = 0
    for id_rec in vRecordingIDs:
        ecg_data, beat_targets, idcs_beat_starts, is_bad_beat = _load_recording(
            id_rec, load_path
        )

        # - High-pass filter `ecg_data`
        if freq_hp is not None:
            ecg_data = ssg.lfilter(b_hp, a_hp, ecg_data, axis=0)

        # - Data samples: Remove part before first beat
        idx_first_beat = idcs_beat_starts[0]
        # - Shift indices in idcs_beat_starts accordingly
        idcs_beat_starts += idx_start - idx_first_beat

        ## -- Append processed data to lists
        signal_list.append(ecg_data[idx_first_beat:])
        idcs_beat_starts_coll += list(idcs_beat_starts)
        targets_coll += list(beat_targets)
        # - Signal quality
        is_bad_beat_coll += list(is_bad_beat)
        # - Recording ID
        rec_ids_coll += beat_targets.size * [int(id_rec)]
        # - Increment idx_start
        idx_start += ecg_data.shape[0] - idx_first_beat

    print("Finished loading data.              ")

    # - Flatten data
    rec_data = np.vstack(signal_list)
    idcs_beat_ends = idcs_beat_starts_coll[1:] + [rec_data.shape[0]]

    # - Generate data frame to index data
    annotations = pd.DataFrame(
        {
            "idx_start": idcs_beat_starts_coll,
            "idx_end": idcs_beat_ends,
            "target": targets_coll,
            "recording": rec_ids_coll,
            "bad_signal": is_bad_beat_coll,
        }
    )
    annotations["is_anomal"] = annotations["target"] != translate_annotations["N"]
    # - Store as smaller integers to save memory
    annotations["target"] = annotations["target"].astype("uint8")
    annotations["recording"] = annotations["recording"].astype("uint8")
    annotations["idx_start"] = annotations["idx_start"].astype("uint32")
    annotations["idx_start"] = annotations["idx_start"].astype("uint32")

    # --Print statistics over targets
    tgt_counts = {
        tgt: np.sum(annotations.target == tgt) for tgt in np.unique(annotations.target)
    }

    # # - Weigh beats by inverse number of occurences of their respective targets
    # dfTgtWeights = {
    #     tgt: 1.0 / count
    #     for tgt, count in tgt_counts
    # }
    # annotations["fWeight"] = [dfTgtWeights[strTgt] for strTgt in annotations.target]

    print("Number of occurences per target: \n", tgt_counts)

    if save_path is not None:
        np.save(Path(save_path) / "recordings.npy", rec_data)
        annotations.to_csv(Path(save_path) / "annotations.csv")

    return annotations, rec_data


def load_from_file(load_path: Union[str, Path]):
    """
    load_from_file - Load ecg signal and beat annotations from .npy and .csv files
    :param load_path:  Path to files.
    :return:
        DataFrame with annotations for each beat (each beat one row)
        2D-array with ecg signal from all recordings ([# time steps x # ecg channels])
    """
    annotations = pd.read_csv(
        Path(load_path) / "annotations.csv",
        index_col=0,
        dtype={
            "idx_start": "uint32",
            "idx_end": "uint32",
            "target": "uint8",
            "recording": "uint8",
            "bad_signal": "bool",
            "is_anomal": "bool",
        },
    )
    rec_data = np.load(os.path.join(load_path, "recordings.npy"))
    print(f"ECG signal and annotaitions have been loaded from {load_path}")
    return annotations, rec_data


### --- Class for providing data to be used in simulations and experiments


class ECGRecordings:
    """
    ECGRecordings - Load ecg signal and annotations from files. Provide data for
                    simulations and experiments and
                    - filter by values in selected categories
                    - select heart beats based on probabilities for the targets
                    - arrange beats in segments which may be continuous and of
                      unique values for selected categories
    """

    default_load_path = save_path
    target_names = target_names
    DT = DT

    def __init__(
        self,
        annotations: Optional[pd.DataFrame] = None,
        ecg_data: Optional[np.ndarray] = None,
        load_path: Union[Path, str, None] = None,
    ):
        # - Load ecg data and annoations
        if ecg_data is None or annotations is None:
            if annotations is not None:
                warn(
                    "ECGRecordings: Only annotations provided. Will load signal and "
                    + "annotations from files."
                )
            elif ecg_data is not None:
                warn(
                    "ECGRecordings: Only ECG signal provided. Will load signal and "
                    + "annotations from files."
                )
            if load_path is None:
                load_path = self.default_load_path
            self.annotations, self.ecg_data = load_from_file(load_path)
        else:
            self.annotations = annotations
            self.ecg_data = ecg_data

        # - Add column to annotations indicating which beats have already been used
        self.annotations["is_used"] = False

    def provide_data(
        self,
        num_beats: Union[int, None],
        include: Dict[str, Any] = {},
        exclude: Dict[str, Any] = {"is_used": True},
        target_probs: Optional[Dict[int, float]] = None,
        continuous_segments: bool = False,
        min_anomal_per_seg: Optional[int] = None,
        match_segments: Union[Set[str], List[str]] = {},
        min_len_segment: int = 1,
        max_len_segment: int = 1,
        remain_unused: bool = False,
        verbose: bool = False,
    ) -> (pd.DataFrame, np.ndarray):
        """
        provide_data - Provide ECG signal and annotation, with heart beats filtered
                       by selectable criteria and, if required, grouped in continuous
                       or non-continuous segments.
        :param num_beats:  Number of beats that should be drawn
        :param include:  Dict of categories and allowed values.
        :param exclude:  Dict of categories and values for which beats are included.
        :param target_probs:  Any of the following:
                               - dict with probabilities for each target. Targets
                                 not mentioned will have probability 0.
                               - string saying "uniform": All targets have same probability.
                               - `None`: Probabilities are proportional to number of beats
                                         for each target.
        :param continuous_segments:  If `True` and `min_len_segment` > 1, heartbeats will
                                     be grouped in continuously recorded segments.
        :param min_anomal_per_seg:  If not None and `continuous_segments` is True and
                                    match_segments does not contain "target", segments
                                    are generated where anomalous segments contain at
                                    least `min_anomal_per_seg` beats of one anomaly.
        :param match_segments:  Categories in which beats of a segment have to share the
                                same values. Currently only the combinations `{"target"}`
                                and `{"target", "recording"}` are supported.
        :param min_len_segment:  Minimum segment length. Default: 1
        :param max_len_segment:  Maximum segment length. Default: 1 (must not be less than `min_len_segment`)
        :param remain_unused:    If `True` do not mark selected heartbeats as 'is_used'.
        :param verbose:  Print detailed output for some configurations
        :return:
            DataFrame with annotations of selected heartbeats
            2D-array with ECG signal for selected heartbeats (shape: #timestes x #channels (=2)).
        """
        # - Filter according to `include` and `exclude` keywords
        annotations = self._filter_data(include, exclude)

        if num_beats is None:
            # - Skip process of drawing beats and arranging them in segments
            selection = annotations
            ids_segment = None
        else:
            # - Make sure values for `min_len_segment` and `max_len_segment` are sensible
            if min_len_segment < 1 or max_len_segment < 1:
                raise ValueError(
                    "ECGRecordings: `min_len_segment` and `max_len_segment` must be at least 1."
                )
            if max_len_segment < min_len_segment:
                raise ValueError(
                    f"ECGRecordings: `min_len_segment` ({max_len_segment}) cannot be greater"
                    + f" than `max_len_segment` ({max_len_segment})."
                )

            if (
                (not continuous_segments) and (not match_segments)
            ) or max_len_segment == 1:
                # - Pick beats randomly accordign to `target_probs` without arranging them in segments.
                beat_indices = _pick_beats(num_beats, annotations, target_probs)
                ids_segment = None
            elif continuous_segments:
                # - Ignore 'recording' in `match_segments`. For continuous segs. recording always matches.
                match_segments = set(match_segments) - {"recording"}
                if match_segments - {"target"}:
                    warn(
                        "ECGRecordings: For continuous segments, matching is currently only "
                        + "supported for 'target' (and per definition enforced for 'recording')."
                    )
                if "target" in match_segments:
                    # - Beats are arranged in continuous segments of matching target classes
                    beat_indices, ids_segment = _pick_cont_segments_sameclass(
                        num_beats=num_beats,
                        annotations=annotations,
                        target_probs=target_probs,
                        min_len_segment=min_len_segment,
                        max_len_segment=max_len_segment,
                    )
                elif min_anomal_per_seg is not None:
                    beat_indices, ids_segment = _pick_new_style_segments(
                        num_beats=num_beats,
                        annotations=annotations,
                        min_anomal_per_seg=min_anomal_per_seg,
                        target_probs=target_probs,
                        min_len_segment=min_len_segment,
                        max_len_segment=max_len_segment,
                        verbose=verbose,
                    )
                else:
                    # - Beats are arranged in continuous segments
                    beat_indices, ids_segment = _pick_cont_segments(
                        num_beats=num_beats,
                        annotations=annotations,
                        target_probs=target_probs,
                        min_len_segment=min_len_segment,
                        max_len_segment=max_len_segment,
                    )
            else:
                # - Beats are in non-continuous segments of matching value for specific categorie(s)
                beat_indices, ids_segment = _pick_category_segments(
                    num_beats=num_beats,
                    annotations=annotations,
                    target_probs=target_probs,
                    match_segments=match_segments,
                    min_len_segment=min_len_segment,
                    max_len_segment=max_len_segment,
                )

            # - Annotations for selected beats
            selection = annotations.loc[beat_indices]

        # - Retrieve corresponding ECG signal
        signal = self.generate_signal(selection)

        # - Adjust idx_start and idx_end columns of selection to match with extracted ECG signal
        beat_sizes = np.array(selection.idx_end - selection.idx_start)
        selection["idx_start_new"] = np.cumsum(np.r_[0, beat_sizes[:-1]])
        selection["idx_end_new"] = np.cumsum(beat_sizes)

        if ids_segment is not None:
            # - Add segment IDs to annotations to be able to distinguish segments
            selection["segment_id"] = ids_segment

        if not remain_unused:
            # - Mark selected beats as used
            self.annotations.loc[selection.index, "is_used"] = True

        return selection, signal

    def generate_signal(self, selection):
        if isinstance(selection, pd.DataFrame):
            annotations = selection
        else:
            # - Choose annotations for selected indices
            annotations = self.annotations.loc[selection]
        # - Indices for selecting ecg signal samples
        indices_ecg = [
            i
            for idx_start, idx_end in zip(annotations.idx_start, annotations.idx_end)
            for i in range(idx_start, idx_end)
        ]
        return self.ecg_data[indices_ecg]

    def generate_target(
        self,
        beat_idcs: Union[pd.Index, ArrayLike],
        map_target: Optional[Dict[int, int]] = None,
        extend: Optional[int] = None,
        boolean_raster: bool = False,
    ) -> np.ndarray:
        """
        generate_target - Generate an array containg the target at each time step
                          for a given sequence of hearbeats.
        :param beat_idcs:   Indices (wrt. self.anomalies.index) of heart beats
        :param map_target:  If not `None`, map original targets to other values.
                            Dict with original targets as keys and new targets as values.
        :param extend:      If not `None`, corresponds to a fixed number of time
                            steps by which a nonzero-target is extended after its
                            end, if it is not followed by another non-zero target.
        :param boolean_raster:  If `True`, return target as 2D boolean raster.
        :return:
            if `boolean_raster`:
                2D-bool-array, columns corresponding to (mapped) anomal targets
                in ascending order (normal beats correspond to all False).
            else:
                1D-int-array of target at each time step.
        """
        if isinstance(beat_idcs, pd.DataFrame):
            annotations = beat_idcs
        else:
            # - Choose annotations for selected indices
            annotations = self.annotations.loc[beat_idcs]

        return generate_target(
            annotations=annotations,
            map_target=map_target,
            extend=extend,
            boolean_raster=boolean_raster,
        )

    def _filter_data(
        self, include: Dict[str, Any] = {}, exclude: Dict[str, Any] = {"is_used": True}
    ) -> pd.DataFrame:
        """
        _filter_data - Return subset of annotations where beats are only included if
                       they match the `include` argument and do not match the `exclude`
                       argument.
        :param include:  Dict of categories and allowed values.
        :param exclude:  Dict of categories and values for which beats are included.
        :return:
            DataFrame of annotations for beats that match filters.
        """
        filtered_annot = self.annotations
        # - Iterate over include categories
        for category, values in include.items():
            filtered_annot = filtered_annot.query(f"{category} == {values}")
        # - Iterate over exclude categories
        for category, values in exclude.items():
            filtered_annot = filtered_annot.query(f"{category} != {values}")
        return filtered_annot


### --- Utility functions for ECGRecordings class


def _pick_beats(
    num_beats: int,
    annotations: pd.DataFrame,
    target_probs: Union[None, str, Dict[int, float]] = None,
) -> List[int]:
    """
    _pick_beats - Randomly pick beats, according to probabilitiy distribution for
                  target class or uniformly.
    :param num_beats:  Number of beats that should be drawn
    :param annotations:  Annotations for set of beats from which should be drawn.
    :param target_probs:  Any of the following:
                           - dict with probabilities for each target. Targets
                             not mentioned will have probability 0.
                           - string saying "uniform": All targets have same probability.
                           - `None`: Probabilities are proportional to number of beats
                                     for each target.
    :return:
        List with indices (corresponding to self.annotations.index) of drawn beats.
    """

    if target_probs is None:
        # - Pick beats at random, with target probabilities proportional to number of respective beats
        return list(np.random.choice(annotations.index, size=num_beats))
    else:
        # - Number of beats for each target class
        beat_counts = _determine_beat_counts(
            num_beats, annotations, target_probs, include_zero_prob=False
        )

        # - List for collecting beat indices
        collected_beats: List[int] = []
        # - Pick beats
        for tgt, counts in beat_counts.items():
            collected_beats += list(_pick_target_beats(tgt, counts, annotations))

        # - Shuffle list of beat indices
        np.random.shuffle(collected_beats)

        return collected_beats


def _pick_new_style_segments(
    num_beats: int,
    annotations: pd.DataFrame,
    min_anomal_per_seg: int,
    target_probs: Dict[int, float],
    min_len_segment: int,
    max_len_segment: int,
    verbose: bool = False,
) -> (List[int], List[int]):
    """
    _pick_new_style_segments - Pick segments that are continuous but whose beats
                               are not necessarily all of the same class. If
                               there are anomalous beats in one segment, there
                               must be at least one anomaly type represented
                               with `min_anomal_per_seg` beats or more.
    :param num_beats:  Number of beats to be picked.
    :param annotations:  Annotations of beats from which to pick.
    :param min_anomal_per_seg:  Minimum number of beats of an anomalous class
                                in an anomalous segment.
    :param target_probs:  Probabilities for targets.
    :param min_len_segment:  Minimum segment length.
    :param max_len_segment:  Maximum segment length.
    :param verbose:  Print more detailed output about progress.
    :return:
        List with indices (corresponding to self.annotations.index) of drawn beats.
            Number of returned beats may be slightly larger than `num_beats`.
        List assigning corresponding segment ID to each index.
    """
    if target_probs is None:
        raise ValueError(
            "ECGRecordings: If `min_anomal_per_seg` is not `None`, `target_probs` "
            + "must not be `None`."
        )

    collected_segments = []
    annotations_remaining = annotations.copy()

    ## -- Anomal segments
    normalize = sum(target_probs.values())
    target_nums_missing = {
        tgt: int(np.round(prob * num_beats) / normalize)
        for tgt, prob in target_probs.items()
    }
    while max(target_nums_missing.values()) > 0:
        if verbose:
            print(f"Missing: {target_nums_missing}")
        # - Ignore recordings that does not contain any of the missing targets
        remaining_tgts = {tgt for tgt, num in target_nums_missing.items() if num > 0}
        for rec in np.unique(annotations_remaining.recording):
            annot_rec = annotations_remaining[annotations_remaining.recording == rec]
            # - Split into parts to be able to remove more beats
            num_parts = 8
            split_size = max(
                int(np.ceil(annot_rec.index.size / num_parts)), min_len_segment
            )
            split_annots = np.split(annot_rec, np.arange(1, num_parts) * split_size)
            for i_part, part in enumerate(split_annots):
                if not remaining_tgts.intersection(part.target):
                    annotations_remaining = annotations_remaining.drop(part.index)
                    if verbose:
                        print(
                            f"...Dropped part {i_part + 1} of recording {rec} (size {split_size})"
                        )
        # - Pick segments
        new_beats, new_segs = _pick_cont_segments(
            num_beats=sum(max(n, 0) for n in target_nums_missing.values()),
            annotations=annotations_remaining,
            target_probs=None,
            min_len_segment=min_len_segment,
            max_len_segment=max_len_segment,
        )
        # - Go over new segments and keep those that match criteria
        new_beats = np.asarray(new_beats)
        new_segs = np.asarray(new_segs)
        for i_seg in np.unique(new_segs):
            beats_seg = new_beats[new_segs == i_seg]
            annot_seg = annotations.loc[beats_seg]
            annot_seg_anom = annot_seg[annot_seg.target != 0]
            tgts, cnts = np.unique(annot_seg.target, return_counts=True)
            __, cnts_anom = np.unique(annot_seg_anom.target, return_counts=True)
            if (
                # Only accept fully normal segments or segs with min. number of anomalies
                (
                    (cnts_anom.size > 0 and np.amax(cnts_anom) >= min_anomal_per_seg)
                    or cnts_anom.size == 0
                )
                # Only accept segments that contain targets of which there are still some missing
                and any(target_nums_missing[tgt] > 0 for tgt in tgts)
            ):
                # - Add segment to list
                collected_segments.append(beats_seg)
                # - Remove beat ids from remaining annotations
                annotations_remaining = annotations_remaining.drop(beats_seg)
                # - Reduce counts of missing beats
                for tgt, cnt in zip(tgts, cnts):
                    target_nums_missing[tgt] -= cnt

    # - Shuffle segments
    np.random.shuffle(collected_segments)
    # - Flatten segments to list with indices of chosen beats
    beat_idcs = [idx for seg in collected_segments for idx in seg]
    # - For each time point, identify the corresponding segment
    segment_ids_full = [
        i_seg for i_seg, seg in enumerate(collected_segments) for _ in range(len(seg))
    ]

    return beat_idcs, segment_ids_full


def _pick_category_segments(
    num_beats: int,
    annotations: pd.DataFrame,
    target_probs: Union[None, str, Dict[int, float]] = None,
    match_segments: Union[Set[str], List[str]] = {},
    min_len_segment: int = 1,
    max_len_segment: int = 1,
) -> (List[int], List[int]):
    """
    _pick_category_segments - Return list of indices such that beats are aranged in
                              segments where for specified categories all beats in
                              a segment share the same values.
    :param num_beats:  Number of beats that should be drawn
    :param annotations:  Annotations for set of beats from which should be drawn.
    :param target_probs:  Any of the following:
                           - dict with probabilities for each target. Targets
                             not mentioned will have probability 0.
                           - string saying "uniform": All targets have same probability.
                           - `None`: Probabilities are proportional to number of beats
                                     for each target.
    :param match_segments:  Categories in which beats of a segment have to share the
                            same values. Currently only the combinations `{"target"}`
                            and `{"target", "recording"}` are supported.
    :param min_len_segment:  Minimum segment length. Default: 1
    :param max_len_segment:  Maximum segment length. Default: 1 (must not be less than `min_len_segment`)
    :return:
        List with indices (corresponding to self.annotations.index) of drawn beats.
            Number of returned beats may be slightly larger than `num_beats`.
        List assigning corresponding segment ID to each index
    """
    # - Number of beats for each target class
    beat_counts = _determine_beat_counts(
        num_beats, annotations, target_probs, include_zero_prob=False
    )

    # - List for collecting segments
    collected_segments: List[np.ndarray] = []

    if set(match_segments) == {"target"}:
        ## -- Segments only need to match in target
        # - Iterate over target classes and number of beats that are to be drawn
        for tgt, counts in beat_counts.items():
            beats: np.ndarray = _pick_target_beats(tgt, counts, annotations)
            # - Separate beats into segments
            seg_lengths = _determine_seg_lengths(
                counts, min_len_segment, max_len_segment
            )
            collected_segments += np.split(beats, np.cumsum(seg_lengths[:-1]))
    elif set(match_segments) == {"target", "recording"}:
        ## -- Segments need to match in target and recording
        # - Iterate over target classes and number of beats that are to be drawn
        for tgt, num_beats_tgt in beat_counts.items():
            # - DataFrame with beats of the chosen target
            available_beats = annotations[annotations.target == tgt]
            # - Recordings containing `tgt` and number of beats
            useable_recordings, recording_sizes = np.unique(
                available_beats.recording, return_counts=True
            )
            # # - Ignore recordings with too little beats
            has_sufficient_beats = recording_sizes > min_len_segment
            useable_recordings = useable_recordings[has_sufficient_beats]
            recording_sizes = recording_sizes[has_sufficient_beats]
            # - Determine how many beats should be drawn from each recording
            beats_per_rec = _relative_counts_min(
                num_total=num_beats_tgt,
                distribution=recording_sizes,
                min_count=min_len_segment,
            )
            beats_per_rec = np.asarray(beats_per_rec)
            # - Iterate over recordings determine segment sizes and draw beats
            for rec, n_beats in zip(useable_recordings, beats_per_rec):
                if n_beats > 0:
                    # - Determine sizes of individual segments
                    seg_sizes = _determine_seg_lengths(
                        n_beats, min_len_segment, max_len_segment
                    )
                    # - Draw beats
                    beat_idcs = np.random.choice(
                        available_beats[available_beats.recording == rec].index,
                        size=np.sum(seg_sizes),  # Might be slightly larger than n_beats
                        replace=False,
                    )
                    # - Split into segments
                    new_segments = np.split(beat_idcs, np.cumsum(seg_sizes[:-1]))
                    collected_segments += list(new_segments)
    else:
        raise ValueError(
            "ECGRecordings: Currently only {`target`} and {`target`, `recording`} "
            + "are supported for 'match_segments' argument."
        )

    # - Shuffle the segments
    np.random.shuffle(collected_segments)
    # - Flatten segments to list with indices of chosen beats
    beat_idcs = [idx for seg in collected_segments for idx in seg]
    # - For each time point, identify the corresponding segment
    segment_ids_full = [
        i_seg for i_seg, seg in enumerate(collected_segments) for _ in range(len(seg))
    ]

    return beat_idcs, segment_ids_full


def _relative_counts_min(num_total: int, distribution: ArrayLike, min_count: int = 0):
    """
    _relative_counts - Return an array of integers that sum up to `num_total`,
                       whose values are greater or equal to `min_count` and
                       apart from that are drawn randomly according to
                       `distribution`. If `num_total` is smaller than
                       `len(distribution) * min_count`, some integers will remain 0.
    """
    # - Determine if any integers have to remain 0
    full_size = len(distribution)
    num_nonzero = min(int(np.floor(num_total / min_count)), full_size)
    is_nonzero = np.ones(full_size, bool)
    if num_nonzero < full_size:
        # - Which entries are zero
        is_nonzero[
            np.random.choice(full_size, size=full_size - num_nonzero, replace=False)
        ] = False
    # - New distribution, taking into account uniform distribution of minimum values
    distro_nonzero = np.asarray(distribution)[is_nonzero]
    num_used = num_nonzero * min_count
    distro_remaining = np.clip(
        distro_nonzero - np.mean(distro_nonzero) * num_used / num_total, 0, None
    )
    # - Draw non-zero integers
    counts_nonzero = _relative_counts(num_total - num_used, distro_remaining)
    counts = np.zeros(full_size, int)
    counts[is_nonzero] = counts_nonzero + min_count
    return counts


def _relative_counts(num_total: int, distribution: ArrayLike):
    """
    _relative_counts - Return an array of integers that sum up to `num_total`,
                       whose values are drawn randomly according to
                       `distribution`.
    """
    # - Normalize distribution
    probs = np.array(distribution) / np.sum(distribution)
    # - Draw samples
    samples = np.random.choice(probs.size, size=num_total, p=probs, replace=True)
    # - Count instances
    idcs, counts = np.unique(samples, return_counts=True)
    # - Include 0-values
    counts_full = np.zeros(probs.size)
    counts_full[idcs] = counts

    return counts_full


def _determine_seg_lengths(
    num_beats_total: int, min_len_segment: int, max_len_segment: int
) -> List[int]:
    """
    _determine_seg_lengths - Determine a list of segment lengths between
                             `min_len_segment` and `max_len_segment` such
                             that they add up to `num_beats_total`. Return
                             segment lengths in a list.
    """
    if num_beats_total <= max_len_segment:
        if min_len_segment > num_beats_total:
            warn(
                f"'num_beats_total' ({num_beats_total}) is smaller than minium"
                + f" ({min_len_segment}), will return minium."
            )
        return [min_len_segment]

    # - Maximum necessary number of segments
    max_size = int(np.ceil(num_beats_total / min_len_segment))
    # - Draw many segments, then determine how many need to be kept
    segments = np.random.randint(min_len_segment, max_len_segment + 1, size=max_size)
    summed_segs = np.cumsum(segments)
    segments = list(segments[summed_segs <= num_beats_total])
    current_num_beats = summed_segs[len(segments) - 1]
    num_missing = num_beats_total - current_num_beats
    if num_missing > 0:
        size_last_seg = max(min_len_segment, num_missing)
        segments.append(size_last_seg)
        num_exceeded = size_last_seg - num_missing
        if num_exceeded > 0:
            warn(f"Total number of beats exceeds `num_beats_total` by {num_exceeded}.")
    return segments


def _pick_target_beats(
    target: int, num_beats: int, annotations: pd.DataFrame
) -> np.ndarray:
    """
    _pick_target_beats - Randomly pick beats of a specific target.

    :param target:  Target class for which beats should be drawnl
    :param num_beats:  Number of beats that should be drawnl
    :param annotations:  Annotations for set of beats from which should be drawn.
    :return:
        1D-int-array with IDs of drawn beats.
    """
    annot_tgt = annotations[annotations.target == target]
    try:
        return np.random.choice(annot_tgt.index, size=num_beats, replace=False)
    except ValueError as e:
        # - If not enough beats are available, warn
        num_available = annot_tgt.index.size
        if num_available < num_beats:
            warn(
                f"ECGRecordings: For target {target}, only {num_available} of "
                + f"{num_beats} heartbeats are available. Generated dataset will "
                + "be shorter than requested."
            )
            return np.asarray(annot_tgt.index)
        else:
            # - If exception has different reason, throw it
            raise e


def _pick_cont_segments(
    num_beats: int,
    annotations: pd.DataFrame,
    target_probs: None = None,
    min_len_segment: int = 1,
    max_len_segment: int = 1,
) -> (List[int], List[int]):
    """
    _pick_cont_segments - Return list of indices such that beats are aranged
                          in continuously recorded segments.
    :param num_beats:  Number of beats that should be drawn
    :param annotations:  Annotations for set of beats from which should be drawn.
    :param target_probs:  Currently, this argument is ignored. probabilities
                          are always proportional to number of beats for each
                          target.
    :param min_len_segment:  Minimum segment length. Default: 1
    :param max_len_segment:  Maximum segment length. Default: 1 (must not be less than `min_len_segment`)
    :return:
        List with indices (corresponding to self.annotations.index) of drawn beats.
            Number of returned beats may be slightly larger than `num_beats`.
        List assigning corresponding segment ID to each index
    """
    if target_probs is not None:
        warn(
            "ECGRecordings: Currently it is not possible to set probability for "
            + "continuous segments if heartbeats within a segment do not have a "
            + "common target class. Any argument for `target_probs` is ignored."
        )

    # - Draw segments
    collected_segments = _pick_cont_segments_inner(
        num_beats, annotations, min_len_segment, max_len_segment
    )

    # - Shuffle the segments
    np.random.shuffle(collected_segments)
    # - Flatten segments to list with indices of chosen beats
    beat_idcs = [idx for seg in collected_segments for idx in seg]
    # - For each time point, identify the corresponding segment
    segment_ids_full = [
        i_seg for i_seg, seg in enumerate(collected_segments) for _ in range(len(seg))
    ]

    return beat_idcs, segment_ids_full


def _pick_cont_segments_sameclass(
    num_beats: int,
    annotations: pd.DataFrame,
    target_probs: Union[None, str, Dict[int, float]] = None,
    min_len_segment: int = 1,
    max_len_segment: int = 1,
) -> (List[int], List[int]):
    """
    _pick_cont_segments_sameclass - Return list of indices such that beats are aranged
                                    in continuously recorded segments of heart beats
                                    with common target class.
    :param num_beats:  Number of beats that should be drawn
    :param annotations:  Annotations for set of beats from which should be drawn.
    :param target_probs:  Any of the following:
                           - dict with probabilities for each target. Targets
                             not mentioned will have probability 0.
                           - string saying "uniform": All targets have same probability.
                           - `None`: Probabilities are proportional to number of beats
                                     for each target.
    :param min_len_segment:  Minimum segment length. Default: 1
    :param max_len_segment:  Maximum segment length. Default: 1 (must not be less than `min_len_segment`)
    :return:
        List with indices (corresponding to self.annotations.index) of drawn beats.
            Number of returned beats may be slightly larger than `num_beats`.
        List assigning corresponding segment ID to each index
    """
    # - Number of beats for each target class
    beat_counts = _determine_beat_counts(
        num_beats=num_beats,
        annotations=annotations,
        target_probs=target_probs,
        include_zero_prob=False,
    )

    # - List for collecting segments
    collected_segments: List[np.ndarray] = []

    # - Iterate over target classes and produce segments for each
    for tgt, num_beats_tgt in beat_counts.items():
        # - DataFrame with beats of the chosen target
        available_beats = annotations[annotations.target == tgt]
        # - Pick segments for current target class
        collected_segments += _pick_cont_segments_inner(
            num_beats_tgt, available_beats, min_len_segment, max_len_segment
        )

    # - Shuffle the segments
    np.random.shuffle(collected_segments)
    # - Unravel segments to list with indices of chosen beats
    beat_idcs = [idx for seg in collected_segments for idx in seg]
    # - For each time point, identify the corresponding segment
    segment_ids_full = [
        i_seg for i_seg, seg in enumerate(collected_segments) for _ in range(len(seg))
    ]

    return beat_idcs, segment_ids_full


def _pick_cont_segments_inner(
    num_beats: int,
    annotations: pd.DataFrame,
    min_len_segment: int = 1,
    max_len_segment: int = 1,
) -> List[List[int]]:
    """
    _pick_cont_segments_inner - Return list of list of indices for subset of
                                beats that does not need to be divided by any
                                further criteria. Indices are aranged in
                                segments of continuously recorded beats.
    :param num_beats:  Number of beats that should be drawn
    :param annotations:  Annotations for set of beats from which should be drawn.
    :param min_len_segment:  Minimum segment length. Default: 1
    :param max_len_segment:  Maximum segment length. Default: 1 (must not be less than `min_len_segment`)
    :return:
        List with indices (corresponding to self.annotations.index) of drawn beats.
        Number of returned beats may be slightly larger than `num_beats`.
    """
    # - Split by recording (continuous segments must be within one recording)
    recordings = np.unique(annotations.recording)
    available_recs_idcs = [
        annotations[annotations.recording == rec].index for rec in recordings
    ]
    # - Split indices for each recording into lists of contiguous beats that are sufficiently long.
    available_segs = [
        list(seg)
        for rec_idcs in available_recs_idcs
        for seg in split_at_discontinuity(rec_idcs)
        if len(seg) >= min_len_segment
    ]
    # - Iteratively choose a list and draw a sub-segment from it
    collected_segments: List[List[int]] = []
    while num_beats > 0:
        if len(available_segs) == 0:
            warn("Cannot produce any more continuous segments of the desired length.")
            break
        # - Pick a sub-list of available indices (use list index to be able to delete list)
        #   Weigh sublist by their size (and normalize probabilities)
        p_sublist = np.array([len(sl) for sl in available_segs], float)
        p_sublist /= np.sum(p_sublist)
        idx_sublist = np.random.choice(len(available_segs), p=p_sublist)
        sublist = available_segs.pop(idx_sublist)
        # - Determine length of next segment
        if len(sublist) <= max_len_segment:
            len_seg = min(len(sublist), num_beats)
            # - Append new segment to list and reduce number of remaining beats
            collected_segments.append(sublist[:len_seg])
            num_beats -= len_seg
            continue
        else:
            try:
                len_seg = np.random.randint(
                    min_len_segment,
                    min(max_len_segment, len(sublist) - min_len_segment),
                )
            except ValueError:
                # - It may happen, that no suitable size can be found. Go on, sublist will not be added again.
                continue
        # - Pick start point such that it is less likely that
        idx_start = np.random.choice(
            [0]  # Take first part of sublist
            + [len(sublist) - len_seg]  # Take final part of sublist
            # ...or take part in middle, such that remainders are sufficiently long
            + list(range(min_len_segment, len(sublist) - len_seg - min_len_segment))
        )
        # - Append new segment to list and reduce number of remaining beats
        collected_segments.append(sublist[idx_start : idx_start + len_seg])
        num_beats -= len_seg
        # - Split remaining part of sublist into two continuous segments
        sl0 = sublist[:idx_start]
        sl1 = sublist[idx_start + len_seg :]
        # - Append these lists to list of available indices if they are long enough
        available_segs += [l for l in (sl0, sl1) if len(l) > max_len_segment]

    return collected_segments


def _determine_beat_counts(
    num_beats: int,
    annotations: pd.DataFrame,
    target_probs: Union[str, Dict[int, float], None],
    include_zero_prob: bool = False,
) -> Dict[int, int]:
    """
    _determine_beat_counts - Based on `target_probs` return a dict with the number
                             of beats to be drawn for each target.
    :param num_beats:  Number of beats that should be drawn
    :param annotations:  Annotations for set of beats from which should be drawn.
    :param target_probs:  Any of the following:
                           - dict with probabilities for each target. Targets
                             not mentioned will have probability 0.
                           - string saying "uniform": All targets have same probability.
                           - `None`: Probabilities are proportional to number of beats
                                     for each target.
    :param include_zero_prob:  If `False`, returned dict will only contain entries for
                               targets with non-zero probabilities.
    :return:
        Dict with number of beats for each class
    """
    # - Arrays of target classes and number of available beats for each class
    all_targets, counts_tgt = np.unique(annotations.target, return_counts=True)

    if target_probs is None:
        counts = np.floor(num_beats * counts_tgt / np.sum(counts_tgt)).astype(int)
        beat_counts = {tgt: int(cnt) for tgt, cnt in zip(all_targets, counts)}
    elif target_probs == "uniform":
        # - Assume uniform distribution
        beats_per_tgt = num_beats // len(all_targets)
        beat_counts = {tgt: beats_per_tgt for tgt in all_targets}
    elif isinstance(target_probs, dict):
        # - Normalize probabilities
        norm = sum(target_probs.values())
        target_probs = {tgt: p / norm for tgt, p in target_probs.items()}
        # - Dict with number of beats per class
        beat_counts = {
            tgt: int(np.round(p * num_beats)) for tgt, p in target_probs.items()
        }
    else:
        raise TypeError("ECGRecordings: Did not understand `target_probs` argument.")

    # - Make sure numbers sum up
    diff_beats = num_beats - sum(beat_counts.values())
    # - Add difference to 0-class if available
    try:
        beat_counts[0] += diff_beats
    except KeyError:
        # If no 0-class, distribute beats randomly
        for tgt in random.choices(list(beat_counts.keys()), k=diff_beats):
            beat_counts[tgt] += 1

    if include_zero_prob:
        # - Assume classes that are not in `beat_counts` yet to have probability 0
        #   (This can be the case if `target_probs` does not cover all targets.)
        for tgt in set(all_targets).difference(beat_counts.keys()):
            beat_counts[tgt] = 0

    return beat_counts


def split_at_discontinuity(seq: Iterable[int], step: int = 1) -> List[np.ndarray]:
    """
    split_at_discontinuity - Split a sequence of integers whenever the difference
                             between two subsequent items is not `step`.
    :param  seq:  Iterable over integers
    :param  step:  Difference between two subsequent elements of `seq` that is
                   considered to be continuous. Default: 1
    :return:
        List of arrays of elements from `seq`, separated whenever elements of `seq`
        do not increase by `step`.
    """
    # - Boolean array indicating where `seq` is not continuous
    is_discontinuous = np.r_[False, np.diff(seq) != 1]
    # - Split seq at discontinuities
    return np.split(seq, np.where(is_discontinuous)[0])


def show_colored_overview(
    annotations,
    targetlist: List[int] = [0, 1, 2, 3, 4, 5, 18],
    include_bad_signal: bool = False,
    recordings: Optional[List[int]] = None,
):

    if not include_bad_signal:
        annotations = annotations[annotations.bad_signal == False]

    if recordings is not None:
        annotations = annotations.query(f"recording == {recordings}")

    if len(targetlist) > 7:
        raise ValueError("Currently, `targetlist` can not have more than 7 elements.")

    lColors = [
        (0, 0, 0),  # Black
        (0.12, 0.46, 0.71),  # Blue
        (1.0, 0.5, 0.05),  # Orange
        (0.17, 0.63, 0.17),  # Green
        (0.84, 0.15, 0.16),  # Red
        (0.58, 0.4, 0.74),  # Purple
        (0.55, 0.34, 0.29),  # Brown
        (0.5, 0.5, 0.5),  # Grey
        (1, 1, 1),  # White
    ]
    dColorMap = {nTgt: tplColor for nTgt, tplColor in zip(targetlist, lColors[:-2])}
    id_other_class = len(targetlist)
    id_beat_ended = id_other_class + 1
    dColorMap[id_other_class] = lColors[-2]
    dColorMap[id_beat_ended] = lColors[-1]

    recs = np.unique(annotations.recording)
    nRecordings = recs.size
    nMaxNumBeats = np.amax(
        [annotations[annotations.recording == rec].shape[0] for rec in recs]
    )
    mColors = np.ones((nRecordings, nMaxNumBeats, 3)) * id_beat_ended
    for iRow, nRec in enumerate(recs):
        # - Categories. `id_other_class` corresponds to no used category
        vnTgt0 = np.array((annotations[annotations.recording == nRec]).target)
        vnTargets = np.ones(vnTgt0.size, int) * id_other_class
        # Beats with categories that are used
        bUse = np.isin(vnTgt0, targetlist)
        vnTargets[bUse] = vnTgt0[bUse]
        mColors[iRow, : vnTgt0.size] = np.array([dColorMap[tgt] for tgt in vnTargets])

    plt.imshow(mColors, aspect=30)


def all_segs_cont(annotations):
    """Test whether all segments in `annotations` are continuous"""
    segments = [
        annotations[annotations.segment_id == i]
        for i in np.unique(annotations.segment_id)
    ]

    def cont_test(segment):
        return all(np.array(segment.idx_start)[1:] == np.array(segment.idx_end)[:-1])

    is_continuous = np.array([cont_test(seg) for seg in segments])
    if not all(is_continuous):
        print(f"Discontinuities in segments {np.where(is_continuous == False)[0]}.")
        return False
    else:
        return True


def generate_target(
    annotations: pd.DataFrame,
    map_target: Optional[Dict[int, int]] = None,
    extend: Optional[int] = None,
    boolean_raster: bool = False,
) -> np.ndarray:
    """
    generate_target - Generate an array containg the target at each time step
                      for a given sequence of hearbeats.
    :param beat_idcs:   Indices (wrt. self.anomalies.index) of heart beats
    :param map_target:  If not `None`, map original targets to other values.
                        Dict with original targets as keys and new targets as values.
    :param extend:      If not `None`, corresponds to a fixed number of time
                        steps by which a nonzero-target is extended after its
                        end, if it is not followed by another non-zero target.
    :param boolean_raster:  If `True`, return target as 2D boolean raster.
    :return:
        if `boolean_raster`:
            2D-bool-array, columns corresponding to (mapped) anomal targets
            in ascending order (normal beats correspond to all False).
        else:
            1D-int-array of target at each time step.
    """

    # - Generate target
    # Number of data samples per beat
    beat_sizes = annotations.idx_end - annotations.idx_start
    if map_target is not None:
        # - Remap target IDs
        beat_targets = [map_target[tgt] for tgt in annotations.target]
    else:
        beat_targets = np.array(annotations.target)
    target = np.repeat(beat_targets, beat_sizes)

    if extend is not None:
        tgt_zero = map_target[0] if map_target is not None else 0
        # - Determine when anomalies end
        anom_end_idcs = np.r_[
            False, np.logical_and(target[1:] == tgt_zero, target[:-1] != tgt_zero)
        ]
        for idx_extd in anom_end_idcs:
            target[idx_extd : idx_extd + extend] = target[idx_extd - 1]

    if not boolean_raster:
        return target

    else:
        num_timesteps = target.size
        if map_target is None:
            anom_targets = np.unique(annotations.target[annotations.target != 0])
            num_anom_targets = anom_targets.size
        else:
            anom_targets = set(map_target.values())
            if 0 in map_target:
                anom_targets -= {map_target[0]}
            num_anom_targets = len(anom_targets)
        # - Convert target to boolean raster, ignore target `normal`
        bool_tgt = np.zeros((num_timesteps, num_anom_targets), bool)
        for idx_tgt, tgt in enumerate(sorted(anom_targets)):
            tgt_is_present = target == tgt
            bool_tgt[tgt_is_present, idx_tgt] = True
        return bool_tgt


# # - New Separation of beats:

# kernel = np.exp(-np.arange(1000) / 30).reshape(-1, 1)
# filtered = fftconvolve(ecg_data, kernel, "full", axes=0)[: ecg_data.shape[0]]
# absdiffs = np.abs(np.diff(filtered, axis=0))
# winnedafd = fftconvolve(absdiffs, win, "full")[38 : 38 + ecg_data.shape[0]]
# maxwafd = np.max(winnedafd, axis=1)
# separations = []
# for st, en in zip(annotations.idx_start.values, annotations.idx_end.values):
#     separations.append(st + np.argmin(maxwafd[st:en]))
