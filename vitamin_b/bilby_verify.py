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
# bilby_raw_loglikes_maybe = result.posterior['log_likelihood'] # note this and the one above are the exact same!
num_samples=bilby_raw_loglikes.shape[0]

# n, bins, patches = plt.hist(x=bilby_raw_loglikes, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
# # _,_,_ = plt.hist(x=bilby_raw_loglikes_maybe, bins='auto', color='red',alpha=0.1, rwidth=0.85)
# plt.xlabel('bilby Loglikes')
# plt.ylabel('Frequency')
# plt.title(f'Bilby Loglikes for {num_samples} samples')
# plt.savefig(f'bilby_loglike_hist_{time.strftime("%Y%m%d-%H%M%S")}.png')

'''HOMEMADE READ-INS'''

test_sampler_file = '/scratch/wiay/matthewd/msci_project/vitamin_b/vitamin_b/test_sets/dynesty_forced_phase_margin/test_dynesty1/dynesty_forced_phase_margin_0.h5py'
# test_waveform_file = '/scratch/wiay/matthewd/msci_project/vitamin_b/vitamin_b/test_sets/dynesty_forced_phase_margin/test_waveforms/dynesty_forced_phase_margin_0.h5py'
hf_sampler = h5py.File(test_sampler_file, 'r')
# hf_waveform = h5py.File(test_waveform_file, 'r')

mass_1_samples = hf_sampler['mass_1_post']
mass_2_samples = hf_sampler['mass_2_post']
luminosity_distance_samples = hf_sampler['luminosity_distance_post']
theta_jn_samples=hf_sampler['theta_jn_post']
psi_samples=hf_sampler['psi_post']
geocent_time_samples=hf_sampler['geocent_time_post']
uufd = hf_sampler['uufd']

ref_geocent_time = params['ref_geocent_time']
duration = params['duration']
sampling_frequency = params['ndata']
start_time = ref_geocent_time-duration/2.0

priors = bilby.gw.prior.BBHPriorDict()
priors['phase'] = bilby.gw.prior.Uniform(name='phase', minimum=bounds['phase_min'], maximum=bounds['phase_max'], boundary='periodic')

'''BILBY RUN'''

ifos = bilby.gw.detector.InterferometerList(['H1'])
for ifo_ind, ifo in enumerate(ifos):
    ifo.set_strain_data_from_frequency_domain_strain(
        uufd[ifo_ind,:], sampling_frequency=sampling_frequency, # might change uufd to its squeezed form then dont need to index first axis!
        duration=duration, start_time = start_time
    )

waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                              reference_frequency=20., minimum_frequency=20.)

waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
        start_time=start_time)

likelihood = bilby.gw.GravitationalWaveTransient(interferometers=ifos, waveform_generator=waveform_generator, phase_marginalization=True,priors=priors)

bilby_loglikes=np.zeros([num_samples])

for i in range(num_samples):

    likelihood_parameters = dict(
        mass_1=mass_1_samples[i], # might want to use placeholders inside tf graph, we'll see
        mass_2=mass_2_samples[i],
        luminosity_distance=luminosity_distance_samples[i],
        geocent_time=geocent_time_samples[i],
        theta_jn=theta_jn_samples[i],
        psi=psi_samples[i],
        ra=fixed_vals['ra'], # option to simplify is to get rid of ra and dec. 
        dec=fixed_vals['dec'], # not flat prior, sinusoid prior (try convert to a space that emulates flat prior somehow)
        phase=fixed_vals['phase'], # can set to any float and it doesn't change the overall value due to phase marginalisation.
        a_1=fixed_vals['a_1'], a_2=fixed_vals['a_2'], tilt_1=fixed_vals['tilt_1'], tilt_2=fixed_vals['tilt_2'], phi_12=fixed_vals['phi_12'], phi_jl=fixed_vals['phi_jl'], # all 6 of these vals are zero
        )     
                

    likelihood.parameters = likelihood_parameters
    bilby_loglike_single = likelihood.log_likelihood()
    bilby_loglikes[i]=bilby_loglike_single


# n, bins, patches = plt.hist(x=bilby_raw_loglikes, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
_,_,_ = plt.hist(x=bilby_loglikes, bins='auto', color='red',alpha=0.7, rwidth=0.85)
plt.xlabel('bilby Loglikes')
plt.ylabel('Frequency')
plt.title(f'Bilby Loglikes for {num_samples} samples')
plt.savefig(f'bilby_loglike_hist_{time.strftime("%Y%m%d-%H%M%S")}.png')



