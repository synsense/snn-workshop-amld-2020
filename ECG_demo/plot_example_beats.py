from matplotlib import pyplot as plt
import seaborn as sns
from rockpool import load_ts_from_file

plt.ion()

dt = 1.0 / 360.0

labels = [
    "Normal beats",
    "Left bundle branch block beats",
    "Right bundle branch block beats",
    "Premature ventricular contractions",
    "Paced beats",
    "Atrial premature beats",
]


def plot_examples():
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))

    plt.subplots_adjust(
        top=0.98, bottom=0.05, left=0.15, right=0.95, hspace=0.5, wspace=0.2
    )

    for i_anom, (ax, lbl) in enumerate(zip(axes.flatten(), labels)):
        ts_anom = load_ts_from_file(f"{i_anom}.npz")
        ts_anom.plot(target=ax)
        ax.set_title(lbl)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude [mV]")
        sns.despine()

    return
