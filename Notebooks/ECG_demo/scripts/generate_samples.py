from pathlib import Path
from rockpool import load_ts_from_file

loadpath = Path(
    "/home/felix/gitlab/Projects/AnomalyDetection/ECG/simulations/realecg/new_style_segs_simmba/19-10-07_16-51-36/test"
)

dt = 1.0 / 360.0
dur = 3.5

# - Load timeseries and annotations
tsecg = load_ts_from_file(loadpath / "external_000.npz")

labels = [
    "Normal beats",
    "Left bundle branch block beats",
    "Right bundle branch block beats",
    "Premature ventricular contractions",
    "Paced beats",
    "Atrial premature beats",
]
start_times = [89.52, 59.75, 268.625, 103.87, 147.57]

for i_anom, t_start in zip([0, 1, 3, 4, 5], start_times):
    ts_anom = tsecg.clip(t_start=t_start, t_stop=t_start + dur).delay(-t_start)
    ts_anom.save(str(i_anom))


# - PVC beats need to be loaded from different sample
loadpath_pvc = Path(
    "/home/felix/hdd/Recorded Data/ECG/experiments/DynapSE/7Dim/updown/partitioned_2d_reservoir/realecg/same_class_segs_long_rec"
)

t_start_pvc = 158.13

tspvc = load_ts_from_file(loadpath_pvc / "dTest_external_0.npz")
tspvc.clip(t_start_pvc, t_stop=t_start_pvc + dur).delay(-t_start_pvc).save("2")
