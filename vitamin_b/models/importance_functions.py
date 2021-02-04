import numpy as np
import time
import os, sys,io
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
import corner
import h5py
from lal import GreenwichMeanSiderealTime
import bilby
import scipy
from scipy.special import logsumexp
import argparse

from .neural_networks import VI_decoder_r2
from .neural_networks import VI_encoder_r1
from .CVAE_model import get_param_index
# from .neural_networks import VI_encoder_q
from .neural_networks import batch_manager
try:
    from .. import gen_benchmark_pe
    from .neural_networks.vae_utils import convert_ra_to_hour_angle
except Exception as e:
    import gen_benchmark_pe
    from models.neural_networks.vae_utils import convert_ra_to_hour_angle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tfd = tfp.distributions
SMALL_CONSTANT = 1e-12 # necessary to prevent the division by zero in many operations 
GAUSS_RANGE = 10.0     # Actual range of truncated gaussian when the ramp is 0

def vit_loglike_creation(params,
                         y_data_processed, 
                         load_dir,
                         norm_sample_single, # str filepath for the h5py file of vitamin samples TODO - automate this for any number of test sets in vit results dir # number of saved vit samples to get loglikes for, max = num_samples in h5py file (currently 20,782)
                         z_batch):

    multi_modal = True

    # USEFUL SIZES
    xsh1 = len(params['inf_pars']) #read in from function input variable from [inf params] value in json
    y_normscale = params['y_normscale']
    if params['by_channel'] == True:
        ysh0 = np.shape(y_data_processed)[0]
        ysh1 = np.shape(y_data_processed)[1]
    else:
        ysh0 = np.shape(y_data_processed)[1]
        ysh1 = np.shape(y_data_processed)[2]
    z_dimension = params['z_dimension']
    n_weights_r1 = params['n_weights_r1']
    n_weights_r2 = params['n_weights_r2']
    n_weights_q = params['n_weights_q']
    n_modes = params['n_modes']
    n_hlayers_r1 = len(params['n_weights_r1'])
    n_hlayers_r2 = len(params['n_weights_r2'])
    n_hlayers_q = len(params['n_weights_q'])
    n_conv_r1 = len(params['n_filters_r1']) if params['n_filters_r1'] != None else None
    n_conv_r2 = len(params['n_filters_r2']) if params['n_filters_r2'] != None else None
    n_conv_q = len(params['n_filters_q'])   if params['n_filters_q'] != None else None
    n_filters_r1 = params['n_filters_r1']
    n_filters_r2 = params['n_filters_r2']
    n_filters_q = params['n_filters_q']
    filter_size_r1 = params['filter_size_r1']
    filter_size_r2 = params['filter_size_r2']
    filter_size_q = params['filter_size_q']
    batch_norm = params['batch_norm']
    ysh_conv_r1 = ysh1
    ysh_conv_r2 = ysh1
    ysh_conv_q = ysh1
    drate = params['drate']
    maxpool_r1 = params['maxpool_r1']
    maxpool_r2 = params['maxpool_r2']
    maxpool_q = params['maxpool_q']
    conv_strides_r1 = params['conv_strides_r1']
    conv_strides_r2 = params['conv_strides_r2']
    conv_strides_q = params['conv_strides_q']
    pool_strides_r1 = params['pool_strides_r1']
    pool_strides_r2 = params['pool_strides_r2']
    pool_strides_q = params['pool_strides_q']
    if n_filters_r1 != None:
        if params['by_channel'] == True:
            num_det = np.shape(y_data_processed)[2]
        else:
            num_det = ysh0
    else:
        num_det = None

    # identify the indices of different sets of physical parameters
    vonmise_mask, vonmise_idx_mask, vonmise_len = get_param_index(params['inf_pars'],params['vonmise_pars'])
    gauss_mask, gauss_idx_mask, gauss_len = get_param_index(params['inf_pars'],params['gauss_pars'])
    sky_mask, sky_idx_mask, sky_len = get_param_index(params['inf_pars'],params['sky_pars'])
    ra_mask, ra_idx_mask, ra_len = get_param_index(params['inf_pars'],['ra'])
    dec_mask, dec_idx_mask, dec_len = get_param_index(params['inf_pars'],['dec'])
    m1_mask, m1_idx_mask, m1_len = get_param_index(params['inf_pars'],['mass_1'])
    m2_mask, m2_idx_mask, m2_len = get_param_index(params['inf_pars'],['mass_2'])
    idx_mask = np.argsort(gauss_idx_mask + vonmise_idx_mask + m1_idx_mask + m2_idx_mask + sky_idx_mask) # + dist_idx_mask)
    masses_len = m1_len + m2_len

    '''
    Graph starts here
    '''

    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        tf.set_random_seed(np.random.randint(0,10))
        SMALL_CONSTANT = 1e-12

        # PLACEHOLDERS
        bs_ph = tf.placeholder(dtype=tf.int64, name="bs_ph")                       # batch size placeholder, only allows integer values of nsamp
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None,params['ndata'], num_det], name="y_ph") # this axis=0 None length allows a variable number of waveforms. I just want one
        samp_ph=tf.placeholder(dtype=tf.float32, shape=[None, xsh1], name="samp_ph") # need to have the data type right otherwise the placeholder wont update from feed dict!!!!

        r2_xzy = VI_decoder_r2.VariationalAutoencoder('VI_decoder_r2', vonmise_mask, gauss_mask, m1_mask, m2_mask, sky_mask, n_input1=z_dimension, 
                                                     n_input2=params['ndata'], n_output=xsh1, n_channels=num_det, n_weights=n_weights_r2, 
                                                     drate=drate, n_filters=n_filters_r2, 
                                                     filter_size=filter_size_r2, maxpool=maxpool_r2)
        r1_zy = VI_encoder_r1.VariationalAutoencoder('VI_encoder_r1', n_input=params['ndata'], n_output=z_dimension, n_channels=num_det, n_weights=n_weights_r1,   # generates params for r1(z|y)
                                                    n_modes=n_modes, drate=drate, n_filters=n_filters_r1, 
                                                    filter_size=filter_size_r1, maxpool=maxpool_r1)

        # GET r1(z|y)
        r1_loc, r1_scale, r1_weight = r1_zy._calc_z_mean_and_sigma(y_ph)
        temp_var_r1 = SMALL_CONSTANT + tf.exp(r1_scale)

        # define the r1(z|y) mixture model
        r1_dist = tfd.MixtureSameFamily(
                          mixture_distribution=tfd.Categorical(logits=r1_weight),
                          components_distribution=tfd.MultivariateNormalDiag(
                          loc=r1_loc,
                          scale_diag=tf.sqrt(temp_var_r1)))
        
        zj_samp=r1_dist.sample() 
        
        reconstruction_xzy = r2_xzy.calc_reconstruction(zj_samp,y_ph)
        
        r2_xzy_mean_gauss = reconstruction_xzy[0]
        r2_xzy_log_sig_sq_gauss = reconstruction_xzy[1]
        r2_xzy_mean_vonmise = reconstruction_xzy[2]
        r2_xzy_log_sig_sq_vonmise = reconstruction_xzy[3]
        r2_xzy_mean_m1 = reconstruction_xzy[4]
        r2_xzy_log_sig_sq_m1 = reconstruction_xzy[5]
        r2_xzy_mean_m2 = reconstruction_xzy[6]
        r2_xzy_log_sig_sq_m2 = reconstruction_xzy[7]
        r2_xzy_mean_sky = reconstruction_xzy[8]
        r2_xzy_log_sig_sq_sky = reconstruction_xzy[9]

        '''
        MASSES - only works for one single OG Sample - shape (1,7)
        '''

        temp_var_r2_m1 = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_m1)     # the m1 variance
        temp_var_r2_m2 = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_m2)     # the m2 variance
        mass_dist = tfd.JointDistributionSequential([
                       tfd.Independent(tfd.TruncatedNormal(r2_xzy_mean_m1,tf.sqrt(temp_var_r2_m1),0,1,validate_args=True,allow_nan_stats=True),reinterpreted_batch_ndims=0),  # m1
            lambda b0: tfd.Independent(tfd.TruncatedNormal(r2_xzy_mean_m2,tf.sqrt(temp_var_r2_m2),0,b0,validate_args=True,allow_nan_stats=True),reinterpreted_batch_ndims=0)],    # m2
            validate_args=True)
        masses_loglike = mass_dist.log_prob((tf.boolean_mask(samp_ph,m1_mask,axis=1),tf.boolean_mask(samp_ph,m2_mask,axis=1))) 
        '''
        TRUNCATED GAUSSIANS
        '''
        
        temp_var_r2_gauss = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_gauss)
        gauss_x = tf.boolean_mask(samp_ph,gauss_mask,axis=1)
        tn = tfd.TruncatedNormal(r2_xzy_mean_gauss,tf.sqrt(temp_var_r2_gauss),0.0,1.0)   # shrink the truncation with the ramp
        trunc_gauss_loglike = tf.reduce_sum(tn.log_prob(gauss_x),axis=1)
        trunc_gauss_loglike = tf.expand_dims(trunc_gauss_loglike, axis=1) # get into shape (batch,1)

        '''
        SKY PARAMS
        '''

        temp_var_r2_sky = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_sky)
        con = tf.reshape(tf.math.reciprocal(temp_var_r2_sky),[bs_ph])   # modelling wrapped scale output as log variance - only 1 concentration parameter for all sky
        loc_xyz = tf.math.l2_normalize(tf.reshape(r2_xzy_mean_sky,[-1,3]),axis=1)    # take the 3 output mean params from r2 and normalse so they are a unit vector
        von_mises_fisher = tfp.distributions.VonMisesFisher(
                      mean_direction=loc_xyz,
                      concentration=con)
        ra_sky = 2*np.pi*tf.reshape(tf.boolean_mask(samp_ph,ra_mask,axis=1),[-1,1])       # convert the scaled 0->1 true RA value back to radians
        dec_sky = np.pi*(tf.reshape(tf.boolean_mask(samp_ph,dec_mask,axis=1),[-1,1]) - 0.5) # convert the scaled 0>1 true dec value back to radians
        xyz_unit = tf.reshape(tf.concat([tf.cos(ra_sky)*tf.cos(dec_sky),tf.sin(ra_sky)*tf.cos(dec_sky),tf.sin(dec_sky)],axis=1),[-1,3])   # construct the true parameter unit vector
        sky_loglike = von_mises_fisher.log_prob(tf.math.l2_normalize(xyz_unit,axis=1))   # normalise it for safety (should already be normalised) and compute the logprob
        sky_loglike = tf.expand_dims(sky_loglike, axis=1)

        '''
        COMBINE LOGLIKES
        '''

        single_loglike=tf.concat([masses_loglike,trunc_gauss_loglike,sky_loglike],axis=1) # get into shape (batch,3)
        single_loglike=tf.reduce_sum(single_loglike, axis=1) # then reduce sum over the axis=1 to get shape (batch) (1d array)


        '''
        CALC EXPECTATION VALUE
        '''

        final_loglike=tf.reduce_logsumexp(single_loglike, axis=0) # unnormalised but it's chill

        '''
        Run Session
        '''

        # VARIABLES LISTS
        var_list_VICI = [var for var in tf.trainable_variables() if var.name.startswith("VI")]

        # INITIALISE AND RUN SESSION
        init = tf.initialize_all_variables()
        session.run(init)
        saver_VICI = tf.train.Saver(var_list_VICI)
        saver_VICI.restore(session,load_dir)

    y_data_test_exp = np.tile(y_data_processed,(z_batch,1))/y_normscale
    y_data_test_exp = y_data_test_exp.reshape(-1,params['ndata'],num_det)

    norm_sample_single_tiled = np.tile(norm_sample_single,(z_batch,1))
    loglike = session.run([final_loglike],feed_dict={samp_ph: norm_sample_single_tiled, bs_ph: z_batch, y_ph: y_data_test_exp})

    return loglike


'''
########################################################################################################
Here is where I will experiment with the bilby stuff
########################################################################################################
'''

########################################################################################################


def bilby_stuff(fixed_vals, params, bounds,
                        #  x_data_test, # this is array like, but might need to make it a dict without other test_set readin
                        vit_loglikes, vit_samples, # need y data, FT to freq dom
                        uufd,

                        # seed=None, outdir='./importance_sampling_results', start_sample=0, end_sample=5000,
                        ):

    '''
    INPUTS:
    - fixed_vals, params, bounds all come from the params.jsons (x3) which are global in the run_vitamin script. 
    # - x_data_test: changing this from a dictionary to simply a 1d array as the dict is redundant # a dictionary from the h5py file with 9 rand params with their final values
    - vit loglikes: 1d array of loglikes of vitamin samples from monte function
    - vit_samples: 2d array (nsamp,nparam) where len axis=0 is the same as len vit loglikes 
    - uufd: fd strain data from test_set as isolated in run_vit.gen_samples parent function
    '''

    ref_geocent_time = params['ref_geocent_time']
    duration = params['duration']
    psd_files = params['psd_files'] # if leave blank = bilby (make it an empty list), note its empty in params json so can leave this
    sampling_frequency = params['ndata']
    inf_pars=params['inf_pars']
    rand_pars=params['rand_pars']


    # masks for param extraction for likelihood evaluation...

    gauss_mask, gauss_idx_mask, gauss_len = get_param_index(params['inf_pars'],params['gauss_pars'])
    ra_mask, ra_idx_mask, ra_len = get_param_index(params['inf_pars'],['ra'])
    dec_mask, dec_idx_mask, dec_len = get_param_index(params['inf_pars'],['dec'])
    m1_mask, m1_idx_mask, m1_len = get_param_index(params['inf_pars'],['mass_1'])
    m2_mask, m2_idx_mask, m2_len = get_param_index(params['inf_pars'],['mass_2'])

    # define the start time of the timeseries
    start_time = ref_geocent_time-duration/2.0 # start time to inject signals

    # choose waveform parameters here, only thing this edits from the input d_test_data variable is the geocent time.
    # pars = fixed_vals
    # # print(f'this is pars1 bro: {pars}')
    # for par_idx, par in enumerate(params['rand_pars']):
    #     if par == 'geocent_time':
    #         pars[par] = x_data_test[par_idx] + ref_geocent_time
    #     else:
    #         pars[par] = x_data_test[par_idx]

    # print(f'this is pars2 bro: {pars}')

    '''
    CHECKPOINT: we now have 'pars' which is a dict that originated as all 15 fixed vals params from json param. This was then overwritten by the ACTUAL
    x data from the h5py file for the 9 rand pars. So we have 15 vals, 6 fixed and 9 the true source params.
    Ask Chris what to do if we want to inject signal to match waveform that we DONT KNOW the true x data for. Hunter suggested use the the values of the
    vitamin posterior with the highest loglike (MLE approx)
    '''

    '''Starting again using gen_real_events as a nice benchmark'''

    injection_parameters=None

        # First, put our "data" created above into a list of intererometers (the order is arbitrary)
    ifos = bilby.gw.detector.InterferometerList(params['det'])
    for ifo_ind, ifo in enumerate(ifos):
        ifo.set_strain_data_from_frequency_domain_strain(uufd[ifo_ind,:],
                                                    sampling_frequency=sampling_frequency,
                                                    duration=duration,
                                                    start_time=start_time)

    prior = bilby.core.prior.PriorDict() # need to give phase prior for phase marginalisation!!!
    prior['phase'] = bilby.gw.prior.Uniform(name='phase', minimum=bounds['phase_min'], maximum=bounds['phase_max'], boundary='periodic')



    # # prior['mass_1'] = bilby.gw.prior.Uniform(name='mass_1', minimum=bounds['mass_1_min'], maximum=bounds['mass_1_max'],unit='$M_{\odot}$')
    # # prior['mass_2'] = bilby.gw.prior.Uniform(name='mass_2', minimum=bounds['mass_2_min'], maximum=bounds['mass_2_max'],unit='$M_{\odot}$')
    # prior['geocent_time'] = bilby.core.prior.Uniform(
    #         minimum=ref_geocent_time + bounds['geocent_time_min'],
    #         maximum=ref_geocent_time + bounds['geocent_time_max'],
    #         name='geocent_time', latex_label='$t_c$', unit='$s$')
    # prior['a_1'] =  0.0
    # prior['a_2'] =  0.0
    # prior['tilt_1'] =  0.0
    # prior['tilt_2'] =  0.0
    # prior['phi_12'] =  0.0
    # prior['phi_jl'] =  0.0
    # prior['dec'] =  -1.2232
    # prior['ra'] =  2.19432
    # prior['theta_jn'] =  1.89694
    # prior['psi'] =  0.532268
    # prior['luminosity_distance'] = 412.066

    # Next create a dictionary of arguments which we pass into the LALSimulation waveform - we specify the waveform approximant here
    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2', # NR model selection, the same one as vit traind on
                              reference_frequency=20., minimum_frequency=20.)

    # Next, create a waveform_generator object. This wraps up some of the jobs of converting between parameters etc
    waveform_generator = bilby.gw.WaveformGenerator(
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters)

    # Finally, create our likelihood, passing in what is needed to get going
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifos, waveform_generator, 
        priors=prior,
        time_marginalization=False, phase_marginalization=True, distance_marginalization=False)

    

    # print(np.ma.masked_array(vit_samples[0], mask=m1_mask*1))

    # print(vit_samples[0][0])

    
    
    # weights = []

    # print(f'vit loglikes {vit_loglikes}')

    # for i in range(number_of_samples): # might need to give it all of them but the ones we dont infer might come from the fixed vals.

    '''
    PSEUDOCODE
    using tensorflow here and utlising the batch functionality to do monte carlo integration (sum then divide)
    the difficulty is it's log space so need to do some type of logsumexp
    Everything up to this point in this bilby function has been preamble and thus the same for any sample or any psi int loop.
    Thus, there is little scope for speeding up by tf. I read in the uufd only once,
    I also need to look at the possibility of keeping the batch dim for nsamp instead but will talk to crhis about this.
    I assume i need to np.tile at some point but we shall see. 
    step 1 - create a tensorflow graph
    step 2 - create uniform dist between 0 and pi
    step 3 - sample from this dist N times (in batch at once)
    step 4 - assign N sets of likelihood parameters in batch
    step 5 - this gives a batch of N loglikes
    step 6 - log sum exp across the N-length batch dimension
    step 7 - divide through by N
    
    For clarity, Im going to do it without tensorflow and see how long it takes.
    goal is to get it in batch using tf in the daughter function then loop the parent function like in the vit loglikes
    ask chris his recomendation for getting this bilby bit into a single tf batch
    '''

    def progress(count, total, suffix=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
        sys.stdout.flush()  # As suggested by Rom Ruben

    # set up full array framework:
    N_psi = 1000 # how many times to sample for psi marginalisation
    number_of_samples = len(vit_loglikes)
    bilby_loglikes=np.zeros([number_of_samples,N_psi])

    print('########################################################################################')
    print(f'#### BILBY LOGLIKELIHOODS: number of vitamin samples looped: {number_of_samples}. psi_batchsize = {N_psi} ####')
    print('########################################################################################')    



    for i in range(number_of_samples):
        progress(i+1,number_of_samples,'')#f'           Calculating Bilby Likelihoods for {number_of_samples} VItamin Samples')# f'Calculating Loglikelihood for {num_samples} VItamin sample(s)')
        psi_samples = np.random.uniform(0,np.pi,N_psi) # overwrites itself each vit_sample 

        for ind, sample in enumerate(psi_samples):
            # print(sample) # check it worked and it does!
            likelihood_parameters = dict(

                # fixed for one samples
                mass_1=vit_samples[i,...][0], # might want to use placeholders inside tf graph, we'll see
                mass_2=vit_samples[i,...][1],
                luminosity_distance=vit_samples[i,...][2],
                geocent_time=vit_samples[i,...][3],
                theta_jn=vit_samples[i,...][4], 
                ra=vit_samples[i,...][5], # option to simplify is to get rid of ra and dec. 
                dec=vit_samples[i,...][6], # not flat prior, sinusoid prior (try convert to a space that emulates flat prior somehow)
                phase=0, # can set to any float and it doesn't change the overall value due to phase marginalisation.
                a_1=fixed_vals['a_1'], a_2=fixed_vals['a_2'], tilt_1=fixed_vals['tilt_1'], tilt_2=fixed_vals['tilt_2'], phi_12=fixed_vals['phi_12'], phi_jl=fixed_vals['phi_jl'], # all 6 of these vals are zero

                # changes N_psi times per sample
                psi=sample,
                )

            '''
            scatter plot of pairs of params where color is bilby likelihood
            '''

            # grid approach for psi is more optimum than random sampling

            likelihood.parameters = likelihood_parameters
            bilby_loglike_single = likelihood.log_likelihood()
            bilby_loglikes[i,ind]=bilby_loglike_single

    # print(bilby_loglikes.shape)

    bilby_loglike_means=logsumexp(bilby_loglikes,axis=1) # need to set axis i think

    # print(bilby_loglike_means.shape)


        
        # mass_1=75,
        # mass_2=40,
        # luminosity_distance=fixed_vals['luminosity_distance'],
        # geocent_time=fixed_vals['geocent_time'],
        # theta_jn=fixed_vals['theta_jn'], 
        # ra=fixed_vals['ra'], 
        # dec=fixed_vals['dec'],

        #rand_pars
        # psi=1, # need to scipy quad this over all psi vals, uniform dist between 0 and pi. just set to 1 now for ease.
        # phase=posterior_dict_old['phase'][i],
        #  need to average over like, not loglike, need to logsumexp.

        # '''
        # Do i need to feed in the other 6 fixed vals too to get all 15 likelihood pars or nah?
        # '''
        # note, all 6 of this spin params are fixed at zero

        

    # print(likelihood_parameters['mass_1'])

    # bilby_loglike = []
    # likelihood.parameters = likelihood_parameters
    # bilby_loglike = likelihood.log_likelihood() # dont want ratio look up. got rid of ratio

    # print(f'bilby_loglikes are {bilby_loglike_means}')

    '''IS starts
    TODO - plot different likes
    TODO - start IS reweighting
    TODO - speed up a lot!!!
    ''' 

    # weight = np.exp(bilby_loglike_single - vitamin_loglikes[i]) # instead of this, do it as 2 arrays with np.subtract to get full array of weights instead of element-wise

    # bilby_loglikes.append(bilby_loglike_single)
    # weights.append(weight)

    return bilby_loglike_means