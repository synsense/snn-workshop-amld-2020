from typing import Optional
import numpy as np


def draw_gaussian(
    num_samples: float,
    mean: float,
    std: float,
    min: Optional[float] = None,
    max: Optional[float] = None,
) -> np.ndarray:
    """
    draw_gaussian - Convenience function for drawing `num_samples` samples from
                    Gaussian distribution with mean `mean` and std. dev. `std`
    """
    samples = (np.random.randn(num_samples) * std + 1) * mean
    if (min is not None) or (max is not None):
        np.clip(samples, min, max)
    return samples


# Relative std. dev. for mismatched parameters
MISMATCH = 0.15
DT = 0.001_389  # Base time step (approximately half the recording time step)
NUM_CPU_CORES = 12  # CPU cores that can be used for simulation with nest backend

# - Size of different reservoir partitions
size_expand = 128  # Size of expansion layer
size_rec_exc = 512  # Size of recurrent layer
size_inh = 128  # Size of inhibitory layer
size_reservoir = size_expand + size_rec_exc + size_inh

# Reservoir neuron parameters
# Time constants
tau_mem_exp_mean = 0.02  # Mean expansion layer neuron time constants
tau_mem_rec_mean = 0.175  # Mean exc. rec. neuron time constants
tau_mem_inh_mean = 0.175  # Mean inhib. neuron time constants
tau_syn_exp_mean = 0.1  # Mean syn. time consts. ext. inp. to inp. layer (exc.)
# tau_syn_inp_rec_mean = 0.4  # Mean syn. time consts. inp. layer to exc. rec. neurs.
tau_syn_rec_mean = 0.4  # Mean syn. time consts. exc. rec. connections
tau_syn_rec_inh_mean = 0.4  # Mean syn. time consts. exc. rec. neurs to inh. neurs.
tau_syn_inh_mean = 0.35  # Mean syn. time consts. for inhib. connections

# Base weights
baseweight_ext_to_inp_exc = 5e-4  # Weight for exc. conns from extern. inp. to inp. lyr
baseweight_ext_to_inp_inh = 5e-4  # Weight for inh. conns from extern. inp. to inp. lyr
baseweight_inp_to_rec = 8e-5  # Weight for exc. conns. from inp. to exc. rec. neur.
baseweight_rec = 1.75e-4  # Weight for exc. recurrent connections
baseweight_rec_to_inh = 8e-5  # Weight for exc. conns. to inhib. neurs.
baseweight_inh = 1e-4  # Weight for inhibitory connections

# Further parameters
thresh_exp_mean = 0.01  # Mean expansion layer neuron spiking threshold
thresh_res_rec_mean = 0.01  # Mean excitatory recurrent neuron spiking threshold
thresh_res_inh_mean = 0.01  # Mean reservoir inhibitory neuron spiking threshold
refractory_exp_mean = 0.001  # Expansion layer neuron refractory period
refractory_res_rec_mean = 0.002  # Mean excitatory recurrent neuron refractory period
refractory_res_inh_mean = 0.002  # Mean reservoir inhibitory neuron refractory period


kwargs_reservoir = dict(dt=DT, v_reset=0, v_rest=0, bias=0)

# - Draw neuron and synapse parameters

# Synaptic time constants
tau_syn_expand = draw_gaussian(size_expand, tau_syn_exp_mean, MISMATCH, min=DT)
tau_syn_rec = draw_gaussian(size_rec_exc, tau_syn_rec_mean, MISMATCH, min=DT)
tau_syn_rec_inh = draw_gaussian(size_inh, tau_syn_rec_inh_mean, MISMATCH, min=DT)
kwargs_reservoir["tau_syn_exc"] = np.r_[tau_syn_expand, tau_syn_rec, tau_syn_rec_inh]
kwargs_reservoir["tau_syn_inh"] = draw_gaussian(
    size_reservoir, tau_syn_inh_mean, MISMATCH, min=DT
)

# - Neuron time constants
tau_mem_expand = draw_gaussian(size_expand, tau_mem_exp_mean, MISMATCH, min=DT)
tau_mem_rec = draw_gaussian(size_rec_exc, tau_mem_rec_mean, MISMATCH, min=DT)
tau_mem_inh = draw_gaussian(size_inh, tau_mem_inh_mean, MISMATCH, min=DT)
kwargs_reservoir["tau_mem"] = np.r_[tau_mem_expand, tau_mem_rec, tau_mem_inh]

# Draw refractory periods and firing thresholds
refractory_exp = draw_gaussian(size_expand, refractory_exp_mean, MISMATCH, min=DT)
refractory_res_rec = draw_gaussian(
    size_rec_exc, refractory_res_rec_mean, MISMATCH, min=DT
)
refractory_res_inh = draw_gaussian(size_inh, refractory_res_inh_mean, MISMATCH, min=DT)
kwargs_reservoir["refractory"] = np.r_[
    refractory_exp, refractory_res_rec, refractory_res_inh
]

thresh_exp = draw_gaussian(size_expand, thresh_exp_mean, MISMATCH)
thresh_res_rec = draw_gaussian(size_rec_exc, thresh_res_rec_mean, MISMATCH)
thresh_res_inh = draw_gaussian(size_inh, thresh_res_inh_mean, MISMATCH)
kwargs_reservoir["v_thresh"] = np.r_[thresh_exp, thresh_res_rec, thresh_res_inh]

np.savez("kwargs_reservoir.npz", **kwargs_reservoir)
