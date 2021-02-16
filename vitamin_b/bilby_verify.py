import bilby
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import h5py

# Raw json loglikes

bilby_json = '/scratch/wiay/matthewd/msci_project/vitamin_b/vitamin_b/test_sets/dynesty_forced_phase_margin/test_dynesty1/dynesty_forced_phase_margin_0_result.json'
result = bilby.result.read_in_result(filename=bilby_json)
bilby_raw_loglikes = result.log_likelihood_evaluations
num_samples=bilby_raw_loglikes.shape[0]
bilby_raw_samples = result.posterior['mass_1']

# n, bins, patches = plt.hist(x=bilby_raw_loglikes, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
# plt.xlabel('bilby Loglikes')
# plt.ylabel('Frequency')
# plt.title(f'Bilby Loglikes for {num_samples} samples')
# plt.savefig(f'bilby_loglike_hist_{time.strftime("%Y%m%d-%H%M%S")}.png')


# uufd

test_sampler_file = '/scratch/wiay/matthewd/msci_project/vitamin_b/vitamin_b/test_sets/dynesty_forced_phase_margin/test_dynesty1/dynesty_forced_phase_margin_0.h5py'
test_waveform_file = '/scratch/wiay/matthewd/msci_project/vitamin_b/vitamin_b/test_sets/dynesty_forced_phase_margin/test_waveforms/dynesty_forced_phase_margin_0.h5py'
hf_sampler = h5py.File(test_sampler_file, 'r')
hf_waveform = h5py.File(test_waveform_file, 'r')





# sampler_post = hf_sampler['mass_1_post']

# print(bilby_raw_samples, sampler_post.value)
