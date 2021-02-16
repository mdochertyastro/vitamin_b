import bilby
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import h5py
import json
import os

'''READ-IN PARAMS FILES'''

# Define default location of the parameters files
params = os.path.join(os.getcwd(), 'params_files', 'params.json')
bounds = os.path.join(os.getcwd(), 'params_files', 'bounds.json')
fixed_vals = os.path.join(os.getcwd(), 'params_files', 'fixed_vals.json')
with open(params, 'r') as fp:
    params = json.load(fp)
with open(bounds, 'r') as fp:
    bounds = json.load(fp)
with open(fixed_vals, 'r') as fp:
    fixed_vals = json.load(fp)


'''BILBY RAW RESULT.JSON'''

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

'''HOMEMADE READ-INS'''

test_sampler_file = '/scratch/wiay/matthewd/msci_project/vitamin_b/vitamin_b/test_sets/dynesty_forced_phase_margin/test_dynesty1/dynesty_forced_phase_margin_0.h5py'
test_waveform_file = '/scratch/wiay/matthewd/msci_project/vitamin_b/vitamin_b/test_sets/dynesty_forced_phase_margin/test_waveforms/dynesty_forced_phase_margin_0.h5py'
hf_sampler = h5py.File(test_sampler_file, 'r')
hf_waveform = h5py.File(test_waveform_file, 'r')

mass_1_samples = hf_sampler['mass_1_post']
mass_2_samples = hf_sampler['mass_2_post']
luminosity_distance_samples = hf_sampler['luminosity_distance_post']
geocent_time_samples=hf_sampler['geocent_time_post']
theta_jn_samples=hf_sampler['theta_jn_post']
psi_samples=hf_sampler['psi_post']

uufd = hf_waveform['uufd']




'''BILBY RUN'''

# ifos = bilby.gw.detector.InterferometerList(['H1'])

# for ifo in ifos:

#     ifo.minimum_frequency = minimum_frequency
#     ifo.maximum_frequency = sampling_frequency/2.

#     ifo.set_strain_data_from_frequency_domain_strain(uufd, sampling_frequency=sampling_frequency,duration=duration, start_time = start_time)

# # Construct the appropriate waveform generator
# waveform_arguments_HM = dict(
#                              reference_frequency=50., minimum_frequency=20.
#                              )

# waveform_generator_HM = bilby.gw.WaveformGenerator(
#     duration=duration, sampling_frequency=sampling_frequency,
#     #frequency_domain_source_model=src.NRSur7dq2_nominal,
#     frequency_domain_source_model=source_gws2.gws_nominal,
#     parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
#     waveform_arguments=waveform_arguments_HM,
#     start_time=start_time)

# # Create the GW likelihood
# likelihood = bilby.gw.GravitationalWaveTransient(
#     interferometers=ifos, waveform_generator=waveform_generator_HM)

# # Now, it is time to determine the new likelihood values
# likelihoods_22 = result.posterior['log_likelihood'] # from the result json file, this is a list of log_likelihoods for each n samples, 1d array with dict key being 'log_likelihood' the overall dict being posterior and the entry being a 1s array len nsamp.
# print(result)
# posterior_dict_22 = result.posterior # contains all param samples and likelihoods as a dict
# number_of_samples = len(likelihoods_22) # nsamp from json 

# print(start_sample, number_of_samples)

# likelihoods_HM = []
# weights = []

# if end_sample >= number_of_samples: print('setting end sample to max sample'); end_sample = number_of_samples

# if start_sample >= number_of_samples:
#     raise ValueError('You are outside of the number of samples')

# for i in range(start_sample,end_sample):

#     likelihood_parameters = dict(
#         mass_1=posterior_dict_22['mass_1'][i],
#         mass_2=posterior_dict_22['mass_2'][i],
#         chi_1=posterior_dict_22['chi_1'][i], chi_2=posterior_dict_22['chi_2'][i],
#         luminosity_distance=posterior_dict_22['luminosity_distance'][i],
#         theta_jn=posterior_dict_22['theta_jn'][i], psi=posterior_dict_22['psi'][i],
#         phase=posterior_dict_22['phase'][i],
#         geocent_time=posterior_dict_22['geocent_time'][i],
#         ra=posterior_dict_22['ra'][i], dec=posterior_dict_22['dec'][i])

#     likelihood.parameters = likelihood_parameters
#     likelihood_HM = likelihood.log_likelihood_ratio()
#     weight = np.exp(likelihood_HM - likelihoods_22[i])

#     likelihoods_HM.append(likelihood_HM)
#     weights.append(weight)

#     print(likelihoods_22[i], likelihood_HM, likelihood_HM - likelihoods_22[i])
#     print('evalution {}/{}'.format(i, number_of_samples))

# array_to_be_saved = np.array([likelihoods_22[start_sample:end_sample],
#                               likelihoods_HM, weights]).T

# np.savetxt(outdir+'/HM_evaluations/s{}e{}.dat'.format(start_sample,end_sample),
#     array_to_be_saved)


