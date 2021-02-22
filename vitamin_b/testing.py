import h5py
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# filename = '/scratch/wiay/matthewd/msci_project/vitamin_b/vitamin_b/vitamin_results/3det_9pars_256Hz/1000samples_20210209-193144.h5py'

# filename = '/scratch/wiay/matthewd/msci_project/vitamin_b/vitamin_b/vitamin_results/3det_9pars_256Hz/10bilby_loglikes_20210209-224922.h5py'

# filename = '/scratch/wiay/matthewd/msci_project/vitamin_b/vitamin_b/vitamin_results/3det_9pars_256Hz/1000bilby_loglikes_20210209-230800.h5py'


# filename = '/scratch/wiay/matthewd/msci_project/vitamin_b/vitamin_b/vitamin_results/3det_9pars_256Hz/1000bilby_loglikes_20210210-033624.h5py'

# filename = '/scratch/wiay/matthewd/msci_project/vitamin_b/vitamin_b/test_sets/all_4_samplers/test_dynesty1/all_4_samplers_0.h5py'

# filename = '/scratch/wiay/matthewd/msci_project/vitamin_b/vitamin_b/vitamin_results/3det_9pars_256Hz/1000bilby_loglikes_20210212-094026.h5py'

filename = '/scratch/wiay/matthewd/msci_project/vitamin_b/vitamin_b/vitamin_results/3det_9pars_256Hz/1000bilby_loglikes_20210212-105004.h5py'

hf = h5py.File(filename, 'r')

# print(hf.keys())

# num_samples=int(hf['dec_post'].shape[0])


# inf_pars = [
#         "mass_1",
#         "mass_2",
#         "luminosity_distance",
#         "geocent_time",
#         "theta_jn",
#         "ra",
#         "dec"
#     ]

# norm_samples=np.zeros([num_samples,len(inf_pars)])
# for i in range(num_samples):
#     for index, name in enumerate(inf_pars):
#         norm_samples[i,index]=hf[f'{name}_post'][i]

# loglikes=np.array(hf['loglikes'])
# loglikes = [ x for x in loglikes if x >= -200 ] 

# print(norm_samples[0,...])

loglikes=hf['bilby_loglikes']
num_samples=loglikes.shape[0]

# # '''
# # HISTOGRAMS OF VIT LOGLIKES
# # '''

# # An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=loglikes, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
plt.xlabel('bilby Loglikes')
plt.ylabel('Frequency')
plt.title(f'Bilby Loglikes for {num_samples} samples at 1,000 psi/phase samples')
# # plt.text(23, 45, r'$\mu=15, b=3$')
# # maxfreq = n.max()
# # # Set a clean upper y-axis limit.
# # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.savefig(f'bilby_loglike_hist_{time.strftime("%Y%m%d-%H%M%S")}.png')

# filename = '/scratch/wiay/matthewd/msci_project/vitamin_b/vitamin_b/test_sets/all_4_samplers/test_dynesty1/all_4_samplers_0_result.json'


# import bilby

# result =bilby.result.read_in_result(filename=filename)

# loglikes = result.log_likelihood_evaluations

# num_samples=loglikes.shape[0]

# n, bins, patches = plt.hist(x=loglikes, bins='auto', color='#0504aa',
#                             alpha=0.7, rwidth=0.85)
# # plt.grid(axis='y', alpha=0.75)
# plt.xlabel('bilby Loglikes')
# plt.ylabel('Frequency')
# plt.title(f'Bilby Loglikes for {num_samples} samples')
# # # plt.text(23, 45, r'$\mu=15, b=3$')
# # # maxfreq = n.max()
# # # # Set a clean upper y-axis limit.
# # # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
# plt.savefig(f'bilby_loglike_hist_{time.strftime("%Y%m%d-%H%M%S")}.png')


# print(loglikes.shape)

# import json

# with open(filename) as f:
#     data = json.load(f)

# print(data.keys())