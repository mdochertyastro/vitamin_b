################################################################################################################
#
# --Variational Inference for gravitational wave parameter estimation--
# 
# Our model takes as input measured signals and infers target images/objects.
#
################################################################################################################

import numpy as np
import time
import os, sys,io
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
import corner
import h5py
from lal import GreenwichMeanSiderealTime

from .neural_networks import VI_decoder_r2
from .neural_networks import VI_encoder_r1
from .neural_networks import VI_encoder_q
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

def load_chunk(input_dir,inf_pars,params,bounds,fixed_vals,load_condor=False):
    """ Function to load more training/testing data

    Parameters
    ----------
    input_dir: str
        Directory where training or testing files are stored
    inf_pars: list
        list of parameters to infer when training ML model
    params: dict
        Dictionary containing parameter values of run
    bounds: dict
        Dictionary containing the allowed bounds of source GW parameters
    fixed_vals: dict
        Dictionary containing the fixed values of GW source parameters
    load_condor: bool
        if True, load test samples rather than training samples

    Returns
    -------
    x_data: array_like
        data source parameter values
    y_data_train: array_like
        data time series 
    """

    # load generated samples back in
    train_files = []
    if type("%s" % input_dir) is str:
        dataLocations = ["%s" % input_dir]
        data={'x_data': [], 'y_data_noisefree': [], 'y_data_noisy': [], 'rand_pars': []}

    if load_condor == True:
        filenames = sorted(os.listdir(dataLocations[0]), key=lambda x: int(x.split('.')[0].split('_')[-1]))
    else:
        filenames = os.listdir(dataLocations[0])

    snrs = []
    for filename in filenames:
        try:
            train_files.append(filename)
        except OSError:
            print('Could not load requested file')
            continue

    train_files_idx = np.arange(len(train_files))[:int(params['load_chunk_size']/1000.0)]
    np.random.shuffle(train_files)
    train_files = np.array(train_files)[train_files_idx]
    for filename in train_files: 
            print('... Loading file -> ' + filename)
            data_temp={'x_data': h5py.File(dataLocations[0]+'/'+filename, 'r')['x_data'][:],
                  'y_data_noisefree': h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_noisefree'][:],
                  'rand_pars': h5py.File(dataLocations[0]+'/'+filename, 'r')['rand_pars'][:]}
            data['x_data'].append(data_temp['x_data'])
            data['y_data_noisefree'].append(np.expand_dims(data_temp['y_data_noisefree'], axis=0))
            data['rand_pars'] = data_temp['rand_pars']


    data['x_data'] = np.concatenate(np.array(data['x_data']), axis=0).squeeze()
    data['y_data_noisefree'] = np.concatenate(np.array(data['y_data_noisefree']), axis=0)

    if load_condor == False:
        # convert ra to hour angle if needed
        data['x_data'] = convert_ra_to_hour_angle(data['x_data'], params, rand_pars=True)

    # normalise the data parameters
    for i,k in enumerate(data_temp['rand_pars']):
        par_min = k.decode('utf-8') + '_min'
        par_max = k.decode('utf-8') + '_max'
        data['x_data'][:,i]=(data['x_data'][:,i] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])
    x_data = data['x_data']
    y_data = data['y_data_noisefree']

    # extract inference parameters
    idx = []
    for k in inf_pars:
        for i,q in enumerate(data['rand_pars']):
            m = q.decode('utf-8')
            if k==m:
                idx.append(i)
    x_data = x_data[:,idx]

    
    # reshape arrays for multi-detector
    y_data_train = y_data
    y_data_train = y_data_train.reshape(y_data_train.shape[0]*y_data_train.shape[1],y_data_train.shape[2]*y_data_train.shape[3])

    # reshape y data into channels last format for convolutional approach
    if params['n_filters_r1'] != None:
        y_data_train_copy = np.zeros((y_data_train.shape[0],params['ndata'],len(params['det'])))

        for i in range(y_data_train.shape[0]):
            for j in range(len(params['det'])):
                idx_range = np.linspace(int(j*params['ndata']),int((j+1)*params['ndata'])-1,num=params['ndata'],dtype=int)
                y_data_train_copy[i,:,j] = y_data_train[i,idx_range]
        y_data_train = y_data_train_copy

    return x_data, y_data_train

def get_param_index(all_pars,pars):
    """ Get the list index of requested source parameter types 
  
    Parameters
    ----------
    all_pars: list
        list of infered parameters
    pars: list
        parameters to get index of

    Returns
    -------
    mask: list
        boolean array of parameter indices
    idx: list
        array of parameter indices
    np.sum(mask): float
        total number of parameter indices
    """
    # identify the indices of wrapped and non-wrapped parameters - clunky code
    mask = []
    idx = []
    
    # loop over inference params
    for i,p in enumerate(all_pars):

        # loop over wrapped params 
        flag = False
        for q in pars:
            if p==q:
                flag = True    # if inf params is a wrapped param set flag
        
        # record the true/false value for this inference param
        if flag==True:
            mask.append(True)
            idx.append(i)
        elif flag==False:
            mask.append(False)
     
    return mask, idx, np.sum(mask)

def run(params, y_data_test, siz_x_data, y_normscale, load_dir):
    """ Function to run a pre-trained tensorflow neural network
    
    Parameters
    ----------
    params: dict
        Dictionary containing parameter values of the run
    y_data_test: array_like
        test sample time series
    siz_x_data: float
        Number of source parameters to infer
    y_normscale: float
        arbitrary normalization factor for time series
    load_dir: str
        location of pre-trained model

    Returns
    -------
    xs: array_like
        predicted posterior samples
    (run_endt-run_startt): float
        total time to make predicted posterior samples
    mode_weights: array_like
        learned Gaussian Mixture Model modal weights
    """
    multi_modal = True

    # USEFUL SIZES
    xsh1 = siz_x_data
    if params['by_channel'] == True:
        ysh0 = np.shape(y_data_test)[0]
        ysh1 = np.shape(y_data_test)[1]
    else:
        ysh0 = np.shape(y_data_test)[1]
        ysh1 = np.shape(y_data_test)[2]
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
            num_det = np.shape(y_data_test)[2]
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

   
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        tf.set_random_seed(np.random.randint(0,10))
        SMALL_CONSTANT = 1e-12

        # PLACEHOLDERS
        bs_ph = tf.placeholder(dtype=tf.int64, name="bs_ph")                       # batch size placeholder
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, params['ndata'], num_det], name="y_ph")

        # LOAD VICI NEURAL NETWORKS
        r2_xzy = VI_decoder_r2.VariationalAutoencoder('VI_decoder_r2', vonmise_mask, gauss_mask, m1_mask, m2_mask, sky_mask, n_input1=z_dimension, 
                                                     n_input2=params['ndata'], n_output=xsh1, n_channels=num_det, n_weights=n_weights_r2, 
                                                     drate=drate, n_filters=n_filters_r2, 
                                                     filter_size=filter_size_r2, maxpool=maxpool_r2)
        r1_zy = VI_encoder_r1.VariationalAutoencoder('VI_encoder_r1', n_input=params['ndata'], n_output=z_dimension, n_channels=num_det, n_weights=n_weights_r1,   # generates params for r1(z|y)
                                                    n_modes=n_modes, drate=drate, n_filters=n_filters_r1, 
                                                    filter_size=filter_size_r1, maxpool=maxpool_r1)
        q_zxy = VI_encoder_q.VariationalAutoencoder('VI_encoder_q', n_input1=xsh1, n_input2=params['ndata'], n_output=z_dimension, 
                                                     n_channels=num_det, n_weights=n_weights_q, drate=drate, 
                                                     n_filters=n_filters_q, filter_size=filter_size_q, maxpool=maxpool_q)

        # reduce the y data size
        y_conv = y_ph

        # GET r1(z|y)
        r1_loc, r1_scale, r1_weight = r1_zy._calc_z_mean_and_sigma(y_conv)
        temp_var_r1 = SMALL_CONSTANT + tf.exp(r1_scale)


        # define the r1(z|y) mixture model
        bimix_gauss = tfd.MixtureSameFamily(
                          mixture_distribution=tfd.Categorical(logits=r1_weight),
                          components_distribution=tfd.MultivariateNormalDiag(
                          loc=r1_loc,
                          scale_diag=tf.sqrt(temp_var_r1)))


        # DRAW FROM r1(z|y)
        r1_zy_samp = bimix_gauss.sample()


        # GET r2(x|z,y) from r1(z|y) samples
        reconstruction_xzy = r2_xzy.calc_reconstruction(r1_zy_samp,y_ph)

        # ugly but needed for now
        # extract the means and variances of the physical parameter distributions
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

        # draw from r2(x|z,y) - the masses
        temp_var_r2_m1 = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_m1)     # the m1 variance
        temp_var_r2_m2 = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_m2)     # the m2 variance
        joint = tfd.JointDistributionSequential([
                       tfd.Independent(tfd.TruncatedNormal(r2_xzy_mean_m1,tf.sqrt(temp_var_r2_m1),0,1,validate_args=True,allow_nan_stats=True),reinterpreted_batch_ndims=0),  # m1
            lambda b0: tfd.Independent(tfd.TruncatedNormal(r2_xzy_mean_m2,tf.sqrt(temp_var_r2_m2),0,b0,validate_args=True,allow_nan_stats=True),reinterpreted_batch_ndims=0)],    # m2
            validate_args=True)
        r2_xzy_samp_masses = tf.transpose(tf.reshape(joint.sample(),[2,-1]))  # sample from the m1.m2 space

        # draw from r2(x|z,y) - the truncated gaussian 
        temp_var_r2_gauss = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_gauss)
        @tf.function    # make this s a tensorflow function
        def truncnorm(idx,output):    # we set up a function that adds the log-likelihoods and also increments the counter
            loc = tf.slice(r2_xzy_mean_gauss,[0,idx],[-1,1])            # take each specific parameter mean using slice
            std = tf.sqrt(tf.slice(temp_var_r2_gauss,[0,idx],[-1,1]))   # take each specific parameter std using slice
            tn = tfd.TruncatedNormal(loc,std,0.0,1.0)                   # define the truncated Gaussian distribution
            return [idx+1, tf.concat([output,tf.reshape(tn.sample(),[bs_ph,1])],axis=1)] # return the updated index and new samples concattenated to the input 
        # we do the loop until we've hit all the truncated gaussian parameters - i starts at 0 and the samples starts with a set of zeros that we cut out later
        idx = tf.constant(0)              # initialise counter
        nsamp = params['n_samples']       # define the number of samples (MUST be a normal int NOT tensor so can't use bs_ph)
        output = tf.zeros([nsamp,1],dtype=tf.float32)    # initialise the output (we cut this first set of zeros out later
        condition = lambda i,output: i<gauss_len         # define the while loop stopping condition
        _,r2_xzy_samp_gauss = tf.while_loop(condition, truncnorm, loop_vars=[idx,output],shape_invariants=[idx.get_shape(), tf.TensorShape([nsamp,None])])
        r2_xzy_samp_gauss = tf.slice(tf.reshape(r2_xzy_samp_gauss,[-1,gauss_len+1]),[0,1],[-1,-1])   # cut out the actual samples - delete the initial vector of zeros

        # draw from r2(x|z,y) - the vonmises part
        temp_var_r2_vonmise = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_vonmise)
        con = tf.reshape(tf.math.reciprocal(temp_var_r2_vonmise),[-1,vonmise_len])   # modelling wrapped scale output as log variance
        von_mises = tfp.distributions.VonMises(loc=2.0*np.pi*r2_xzy_mean_vonmise, concentration=con)
        r2_xzy_samp_vonmise = tf.reshape(tf.math.floormod(von_mises.sample(),(2.0*np.pi))/(2.0*np.pi),[-1,vonmise_len])   # sample from the von mises distribution and shift and scale from -pi-pi to 0-1

        if sky_len>0:
            # draw from r2(x|z,y) - the von mises Fisher 
            temp_var_r2_sky = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_sky)
            con = tf.reshape(tf.math.reciprocal(temp_var_r2_sky),[bs_ph])   # modelling wrapped scale output as log variance - only 1 concentration parameter for all sky
            von_mises_fisher = tfp.distributions.VonMisesFisher(
                              mean_direction=tf.math.l2_normalize(tf.reshape(r2_xzy_mean_sky,[bs_ph,3]),axis=1),
                              concentration=con)   # define p_vm(2*pi*mu,con=1/sig^2)
            xyz = tf.reshape(von_mises_fisher.sample(),[bs_ph,3])          # sample the distribution
            samp_ra = tf.math.floormod(tf.atan2(tf.slice(xyz,[0,1],[-1,1]),tf.slice(xyz,[0,0],[-1,1])),2.0*np.pi)/(2.0*np.pi)   # convert to the rescaled 0->1 RA from the unit vector
            samp_dec = (tf.asin(tf.slice(xyz,[0,2],[-1,1])) + 0.5*np.pi)/np.pi                       # convert to the rescaled 0->1 dec from the unit vector
            r2_xzy_samp_sky = tf.reshape(tf.concat([samp_ra,samp_dec],axis=1),[bs_ph,2])             # group the sky samples
        else:
            r2_xzy_samp_sky = tf.zeros([bs_ph,0], tf.float32)

        # combine the samples
        r2_xzy_samp = tf.concat([r2_xzy_samp_gauss,r2_xzy_samp_vonmise,r2_xzy_samp_masses,r2_xzy_samp_sky],axis=1)
        r2_xzy_samp = tf.gather(r2_xzy_samp,tf.constant(idx_mask),axis=1)
        

        # VARIABLES LISTS
        var_list_VICI = [var for var in tf.trainable_variables() if var.name.startswith("VI")]

        # INITIALISE AND RUN SESSION
        init = tf.initialize_all_variables()
        session.run(init)
        saver_VICI = tf.train.Saver(tf.global_variables())
        saver_VICI.restore(session,load_dir)

    # ESTIMATE TEST SET RECONSTRUCTION PER-PIXEL APPROXIMATE MARGINAL LIKELIHOOD and draw from q(x|y)
    ns = params['n_samples'] # number of samples to save per reconstruction 

    y_data_test_exp = np.tile(y_data_test,(ns,1))/y_normscale
    y_data_test_exp = y_data_test_exp.reshape(-1,params['ndata'],num_det)
    run_startt = time.time()
    xs, mode_weights = session.run([r2_xzy_samp,r1_weight],feed_dict={bs_ph:ns,y_ph:y_data_test_exp})
    run_endt = time.time()

    run_startt = time.time()
    xs, mode_weights = session.run([r2_xzy_samp,r1_weight],feed_dict={bs_ph:ns,y_ph:y_data_test_exp})
    run_endt = time.time()

    return xs, (run_endt - run_startt), mode_weights

def train(params, x_data, y_data, x_data_test, y_data_test, y_data_test_noisefree, y_normscale, save_dir, truth_test, bounds, fixed_vals, posterior_truth_test,snrs_test=None):    
    """ Main function to train tensorflow model
    
    Parameters
    ----------
    params: dict
        Dictionary containing parameter values of run
    x_data: array_like
        array containing training source parameter values
    x_data_test: array_like
        array containing testing source parameter values
    y_data_test: array_like
        array containing noisy testing time series
    y_data_test_noisefree: array_like
        array containing noisefree testing time series
    y_normscale: float
        arbitrary normalization factor for time series
    save_dir: str
        location to save trained tensorflow model
    truth_test: array_like TODO: this is redundant, must be removed ...
        array containing testing source parameter values
    bounds: dict
        allowed bounds of GW source parameters
    fixed_vals: dict
        fixed values of GW source parameters
    posterior_truth_test: array_like
        posterior from test Bayesian sampler analysis 
    """

    # if True, do multi-modal
    multi_modal = True

    # USEFUL SIZES
    xsh = np.shape(x_data)
    ysh = np.shape(y_data)[1]
    z_dimension = params['z_dimension']
    bs = params['batch_size']
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
    maxpool_r1 = params['maxpool_r1']
    maxpool_r2 = params['maxpool_r2']
    maxpool_q = params['maxpool_q']
    conv_strides_r1 = params['conv_strides_r1']
    conv_strides_r2 = params['conv_strides_r2']
    conv_strides_q = params['conv_strides_q']
    pool_strides_r1 = params['pool_strides_r1']
    pool_strides_r2 = params['pool_strides_r2']
    pool_strides_q = params['pool_strides_q']
    batch_norm = params['batch_norm']
    ysh_conv_r1 = int(ysh)
    ysh_conv_r2 = int(ysh)
    ysh_conv_q = int(ysh)
    drate = params['drate']
    ramp_start = params['ramp_start']
    ramp_end = params['ramp_end']
    num_det = len(params['det'])

    # identify the indices of different sets of physical parameters
    vonmise_mask, vonmise_idx_mask, vonmise_len = get_param_index(params['inf_pars'],params['vonmise_pars'])
    gauss_mask, gauss_idx_mask, gauss_len = get_param_index(params['inf_pars'],params['gauss_pars'])
    sky_mask, sky_idx_mask, sky_len = get_param_index(params['inf_pars'],params['sky_pars'])
    ra_mask, ra_idx_mask, ra_len = get_param_index(params['inf_pars'],['ra'])
    dec_mask, dec_idx_mask, dec_len = get_param_index(params['inf_pars'],['dec'])
    m1_mask, m1_idx_mask, m1_len = get_param_index(params['inf_pars'],['mass_1'])
    m2_mask, m2_idx_mask, m2_len = get_param_index(params['inf_pars'],['mass_2'])
    idx_mask = np.argsort(gauss_idx_mask + vonmise_idx_mask + m1_idx_mask + m2_idx_mask + sky_idx_mask) # + dist_idx_mask)

    # print(vonmise_mask, vonmise_idx_mask, params['inf_pars'])
    # exit()

    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():

        # PLACE HOLDERS
        bs_ph = tf.placeholder(dtype=tf.int64, name="bs_ph")                       # batch size placeholder
        x_ph = tf.placeholder(dtype=tf.float32, shape=[None, xsh[1]], name="x_ph") # params placeholder
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, params['ndata'], num_det], name="y_ph")
        ramp = tf.placeholder(dtype=tf.float32)    # the ramp to slowly increase the KL contribution

        # LOAD VICI NEURAL NETWORKS
        r1_zy = VI_encoder_r1.VariationalAutoencoder('VI_encoder_r1', n_input=params['ndata'], n_output=z_dimension, n_channels=num_det, n_weights=n_weights_r1,   # generates params for r1(z|y)
                                                    n_modes=n_modes, drate=drate, n_filters=n_filters_r1, 
                                                    filter_size=filter_size_r1, maxpool=maxpool_r1)
        r2_xzy = VI_decoder_r2.VariationalAutoencoder('VI_decoder_r2', vonmise_mask, gauss_mask, m1_mask, m2_mask, sky_mask, n_input1=z_dimension, 
                                                     n_input2=params['ndata'], n_output=xsh[1], n_channels=num_det, n_weights=n_weights_r2, 
                                                     drate=drate, n_filters=n_filters_r2, 
                                                     filter_size=filter_size_r2, maxpool=maxpool_r2)
        q_zxy = VI_encoder_q.VariationalAutoencoder('VI_encoder_q', n_input1=xsh[1], n_input2=params['ndata'], n_output=z_dimension, 
                                                     n_channels=num_det, n_weights=n_weights_q, drate=drate, 
                                                     n_filters=n_filters_q, filter_size=filter_size_q, maxpool=maxpool_q) 
        tf.set_random_seed(np.random.randint(0,10))

        # reduce the y data size
        y_conv = y_ph

        # GET r1(z|y)
        # run inverse autoencoder to generate mean and logvar of z given y data - these are the parameters for r1(z|y)
        r1_loc, r1_scale, r1_weight = r1_zy._calc_z_mean_and_sigma(y_conv)
        temp_var_r1 = SMALL_CONSTANT + tf.exp(r1_scale)

        
        # define the r1(z|y) mixture model
        bimix_gauss = tfd.MixtureSameFamily(
                          mixture_distribution=tfd.Categorical(logits=r1_weight),
                          components_distribution=tfd.MultivariateNormalDiag(
                          loc=r1_loc,
                          scale_diag=tf.sqrt(temp_var_r1)))


        # DRAW FROM r1(z|y) - given the Gaussian parameters generate z samples
        r1_zy_samp = bimix_gauss.sample()        
        
        # GET q(z|x,y)
        q_zxy_mean, q_zxy_log_sig_sq = q_zxy._calc_z_mean_and_sigma(x_ph,y_conv)

        # DRAW FROM q(z|x,y)
        temp_var_q = SMALL_CONSTANT + tf.exp(q_zxy_log_sig_sq)
        mvn_q = tfp.distributions.MultivariateNormalDiag(
                          loc=q_zxy_mean,
                          scale_diag=tf.sqrt(temp_var_q))
        q_zxy_samp = mvn_q.sample()  
       
        # GET r2(x|z,y)
        eps = tf.random.normal([bs_ph, params['ndata'], num_det], 0, 1., dtype=tf.float32)
        y_ph_ramp = tf.add(tf.multiply(ramp,y_conv), tf.multiply((1.0-ramp), eps))
        reconstruction_xzy = r2_xzy.calc_reconstruction(q_zxy_samp,y_ph_ramp)

        # ugly but required for now - unpack the r2 output params
        r2_xzy_mean_gauss = reconstruction_xzy[0]           # truncated gaussian mean
        r2_xzy_log_sig_sq_gauss = reconstruction_xzy[1]     # truncated gaussian log var
        r2_xzy_mean_vonmise = reconstruction_xzy[2]         # vonmises means
        r2_xzy_log_sig_sq_vonmise = reconstruction_xzy[3]   # vonmises log var
        r2_xzy_mean_m1 = reconstruction_xzy[4]              # m1 mean
        r2_xzy_log_sig_sq_m1 = reconstruction_xzy[5]        # m1 var
        r2_xzy_mean_m2 = reconstruction_xzy[6]              # m2 mean (m2 will be conditional on m1)
        r2_xzy_log_sig_sq_m2 = reconstruction_xzy[7]        # m2 log var (m2 will be conditional on m1)
        r2_xzy_mean_sky = reconstruction_xzy[8]             # sky mean unit vector (3D)
        r2_xzy_log_sig_sq_sky = reconstruction_xzy[9]       # sky log var (1D)

        # COST FROM RECONSTRUCTION - the masses
        # this sets up a joint distribution on m1 and m2 with m2 being conditional on m1
        # the ramp eveolves the truncation boundaries from far away to 0->1 for m1 and 0->m1 for m2
        if m1_len>0 and m2_len>0:
            temp_var_r2_m1 = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_m1)     # the safe r2 variance
            temp_var_r2_m2 = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_m2)
            joint = tfd.JointDistributionSequential([    # shrink the truncation with the ramp
                       tfd.Independent(tfd.TruncatedNormal(r2_xzy_mean_m1,tf.sqrt(temp_var_r2_m1),-GAUSS_RANGE*(1.0-ramp),GAUSS_RANGE*(1.0-ramp) + 1.0),reinterpreted_batch_ndims=0),  # m1
                lambda b0: tfd.Independent(tfd.TruncatedNormal(r2_xzy_mean_m2,tf.sqrt(temp_var_r2_m2),-GAUSS_RANGE*(1.0-ramp),GAUSS_RANGE*(1.0-ramp) + ramp*b0),reinterpreted_batch_ndims=0)],    # m2
            )
            reconstr_loss_masses = joint.log_prob((tf.boolean_mask(x_ph,m1_mask,axis=1),tf.boolean_mask(x_ph,m2_mask,axis=1)))

        # COST FROM RECONSTRUCTION - Truncated Gaussian parts
        # this sets up a loop over uncorreltaed truncated Gaussians 
        # the ramp evolves the boundaries from far away to 0->1 
        if gauss_len>0:
            temp_var_r2_gauss = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_gauss)
            gauss_x = tf.boolean_mask(x_ph,gauss_mask,axis=1)
            @tf.function
            def truncnorm(i,lp):    # we set up a function that adds the log-likelihoods and also increments the counter
                loc = tf.slice(r2_xzy_mean_gauss,[0,i],[-1,1])
                std = tf.sqrt(tf.slice(temp_var_r2_gauss,[0,i],[-1,1]))
                pos = tf.slice(gauss_x,[0,i],[-1,1])  
                tn = tfd.TruncatedNormal(loc,std,-GAUSS_RANGE*(1.0-ramp),GAUSS_RANGE*(1.0-ramp) + 1.0)   # shrink the truncation with the ramp
                return [i+1, lp + tn.log_prob(pos)]
            # we do the loop until we've hit all the truncated gaussian parameters - i starts at 0 and the logprob starts at 0 
            _,reconstr_loss_gauss = tf.while_loop(lambda i,reconstr_loss_gauss: i<gauss_len, truncnorm, [0,tf.zeros([bs_ph],dtype=tf.dtypes.float32)])

        # COST FROM RECONSTRUCTION - Von Mises parts for single parameters that wrap over 2pi
        if vonmise_len>0:
            temp_var_r2_vonmise = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_vonmise)
            # print(temp_var_r2_vonmise.shape)
            con_vonmise = tf.reshape(tf.math.reciprocal(temp_var_r2_vonmise),[bs_ph,vonmise_len])   # modelling wrapped scale output as log variance - convert to concentration
            von_mises = tfp.distributions.VonMises(
                          loc=2.0*np.pi*tf.reshape(r2_xzy_mean_vonmise,[bs_ph,vonmise_len]),   # remap 0>1 mean onto 0->2pi range
                          concentration=con_vonmise)
            reconstr_loss_vonmise = tf.reduce_sum(von_mises.log_prob(2.0*np.pi*tf.reshape(tf.boolean_mask(x_ph,vonmise_mask,axis=1),[-1,vonmise_len])),axis=1)   # 2pi is the von mises input range

            # computing Gaussian likelihood for von mises parameters to be faded away with the ramp
            gauss_vonmises = tfd.TruncatedNormal(r2_xzy_mean_vonmise,
                                                 tf.sqrt(temp_var_r2_vonmise),
                                                 -GAUSS_RANGE*(1.0-ramp),GAUSS_RANGE*(1.0-ramp) + 1.0)
            reconstr_loss_gauss_vonmise = tf.reduce_sum(tf.reshape(gauss_vonmises.log_prob(tf.boolean_mask(x_ph,vonmise_mask,axis=1)),[-1,vonmise_len]),axis=1)        
            reconstr_loss_vonmise = ramp*reconstr_loss_vonmise + (1.0-ramp)*reconstr_loss_gauss_vonmise    # start with a Gaussian model and fade in the true vonmises
        else:
            reconstr_loss_vonmise = 0.0
        tf.debugging.check_numerics(reconstr_loss_vonmise, message='reconstr_loss_vonmise is nan or inf')
        
        
        # if vonmise_len>0:
        #     temp_var_r2_vonmise = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_vonmise)
        #     con = tf.reshape(tf.math.reciprocal(temp_var_r2_vonmise),[-1,vonmise_len])   # modelling wrapped scale output as log variance - convert to concentration
        #     von_mises = tfp.distributions.VonMises(
        #                   loc=2.0*np.pi*(tf.reshape(r2_xzy_mean_vonmise,[-1,vonmise_len])-0.5),   # remap 0>1 mean onto -pi->pi range
        #                   concentration=con)
        #     reconstr_loss_vonmise = tf.reduce_sum(von_mises.log_prob(2.0*np.pi*tf.reshape(tf.boolean_mask(x_ph,vonmise_mask,axis=1),[-1,vonmise_len])),axis=1)   # 2pi is the von mises input range
            
            # reconstr_loss_vonmise = reconstr_loss_vonmise[:,0] + reconstr_loss_vonmise[:,1]

            # computing Gaussian likelihood for von mises parameters to be faded away with the ramp
        #     gauss_vonmises = tfp.distributions.Mul2tivariateNormalDiag(
        #                  loc=r2_xzy_mean_vonmise,
        #                  scale_diag=tf.sqrt(temp_var_r2_vonmise))
        #     reconstr_loss_gauss_vonmise = gauss_vonmises.log_prob(tf.boolean_mask(x_ph,vonmise_mask,axis=1))        
        #     reconstr_loss_vonmise = ramp*reconstr_loss_vonmise + (1.0-ramp)*reconstr_loss_gauss_vonmise    # start with a Gaussian model and fade in the true vonmises
        # else:
        #     reconstr_loss_vonmise = 0.0

        # COST FROM RECONSTRUCTION - Von Mises Fisher (sky) parts
        if sky_len>0:
            temp_var_r2_sky = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_sky)
            con = tf.reshape(tf.math.reciprocal(temp_var_r2_sky),[bs_ph])   # modelling wrapped scale output as log variance - only 1 concentration parameter for all sky
            loc_xyz = tf.math.l2_normalize(tf.reshape(r2_xzy_mean_sky,[-1,3]),axis=1)    # take the 3 output mean params from r2 and normalse so they are a unit vector
            von_mises_fisher = tfp.distributions.VonMisesFisher(
                          mean_direction=loc_xyz,
                          concentration=con)
            ra_sky = 2*np.pi*tf.reshape(tf.boolean_mask(x_ph,ra_mask,axis=1),[-1,1])       # convert the scaled 0->1 true RA value back to radians
            dec_sky = np.pi*(tf.reshape(tf.boolean_mask(x_ph,dec_mask,axis=1),[-1,1]) - 0.5) # convert the scaled 0>1 true dec value back to radians
            xyz_unit = tf.reshape(tf.concat([tf.cos(ra_sky)*tf.cos(dec_sky),tf.sin(ra_sky)*tf.cos(dec_sky),tf.sin(dec_sky)],axis=1),[-1,3])   # construct the true parameter unit vector
            reconstr_loss_sky = von_mises_fisher.log_prob(tf.math.l2_normalize(xyz_unit,axis=1))   # normalise it for safety (should already be normalised) and compute the logprob

            # computing Gaussian likelihood for von mises Fisher (sky) parameters to be faded away with the ramp
            mean_ra = tf.math.floormod(tf.atan2(tf.slice(loc_xyz,[0,1],[-1,1]),tf.slice(loc_xyz,[0,0],[-1,1])),2.0*np.pi)/(2.0*np.pi)    # convert the unit vector to scaled 0->1 RA 
            mean_dec = (tf.asin(tf.slice(loc_xyz,[0,2],[-1,1])) + 0.5*np.pi)/np.pi        # convert the unit vector to scaled 0->1 dec
            mean_sky = tf.reshape(tf.concat([mean_ra,mean_dec],axis=1),[bs_ph,2])        # package up the scaled RA and dec 
            gauss_sky = tfp.distributions.MultivariateNormalDiag(
                         loc=mean_sky,
                         scale_diag=tf.concat([tf.sqrt(temp_var_r2_sky),tf.sqrt(temp_var_r2_sky)],axis=1))   # use the same 1D concentration parameter for both RA and dec dimensions
            reconstr_loss_gauss_sky = gauss_sky.log_prob(tf.boolean_mask(x_ph,sky_mask,axis=1))     # compute the logprob at the true sky location
            reconstr_loss_sky = ramp*reconstr_loss_sky + (1.0-ramp)*reconstr_loss_gauss_sky   # start with a Gaussian model and fade in the true vonmises Fisher
        else:
            reconstr_loss_sky = 0

        cost_R = -1.0*tf.reduce_mean(reconstr_loss_gauss + reconstr_loss_vonmise + reconstr_loss_masses + reconstr_loss_sky)
        r2_xzy_mean = tf.gather(tf.concat([r2_xzy_mean_gauss,r2_xzy_mean_vonmise,r2_xzy_mean_m1,r2_xzy_mean_m2,r2_xzy_mean_sky],axis=1),tf.constant(idx_mask),axis=1)      # put the elements back in order
        r2_xzy_scale = tf.gather(tf.concat([r2_xzy_log_sig_sq_gauss,r2_xzy_log_sig_sq_vonmise,r2_xzy_log_sig_sq_m1,r2_xzy_log_sig_sq_m2,r2_xzy_log_sig_sq_sky],axis=1),tf.constant(idx_mask),axis=1)   # put the elements back in order
        r2_xzy_mean = tf.gather(tf.concat([r2_xzy_mean_gauss,r2_xzy_mean_vonmise,r2_xzy_mean_m1,r2_xzy_mean_m2],axis=1),tf.constant(idx_mask),axis=1)      # put the elements back in order
        r2_xzy_scale = tf.gather(tf.concat([r2_xzy_log_sig_sq_gauss,r2_xzy_log_sig_sq_vonmise,r2_xzy_log_sig_sq_m1,r2_xzy_log_sig_sq_m2],axis=1),tf.constant(idx_mask),axis=1)
       
        log_q_q = mvn_q.log_prob(q_zxy_samp)
        log_r1_q = bimix_gauss.log_prob(q_zxy_samp)   # evaluate the log prob of r1 at the q samples
        KL = tf.reduce_mean(log_q_q - log_r1_q)      # average over batch

        # THE VICI COST FUNCTION
        COST = cost_R + ramp*KL #+ L1_weight_reg)

        # VARIABLES LISTS
        var_list_VICI = [var for var in tf.trainable_variables() if var.name.startswith("VI")]
        
        optimizer = tf.train.AdamOptimizer(params['initial_training_rate'])
        minimize = optimizer.minimize(COST,var_list = var_list_VICI)
 
        # INITIALISE AND RUN SESSION
        init = tf.global_variables_initializer()
        session.run(init)
        saver = tf.train.Saver()

    print()
    print('... Training Inference Model')    
    print()
    # START OPTIMISATION OF OELBO
    indices_generator = batch_manager.SequentialIndexer(params['batch_size'], xsh[0])
    plotdata = []

    # Convert right ascension to hour angle
    x_data_test_hour_angle = np.copy(x_data_test)
    x_data_test_hour_angle = convert_ra_to_hour_angle(x_data_test_hour_angle, params)

    load_chunk_it = 1
    for i in range(params['num_iterations']):

        next_indices = indices_generator.next_indices()

        # if load chunks true, load in data by chunks
        if params['load_by_chunks'] == True and i == int(params['load_iteration']*load_chunk_it):
            x_data, y_data = load_chunk(params['train_set_dir'],params['inf_pars'],params,bounds,fixed_vals)
            load_chunk_it += 1

        # Make noise realizations and add to training data
        next_x_data = x_data[next_indices,:]
        if n_conv_r1 != None:
            next_y_data = y_data[next_indices,:] + np.random.normal(0,1,size=(params['batch_size'],int(params['ndata']),len(params['det'])))
        else:
            next_y_data = y_data[next_indices,:] + np.random.normal(0,1,size=(params['batch_size'],int(params['ndata']*len(params['det']))))
        next_y_data /= y_normscale  # required for fast convergence

        if params['by_channel'] == False:
            next_y_data_new = [] 
            for sig in next_y_data:
                next_y_data_new.append(sig.T)
            next_y_data = np.array(next_y_data_new)
            del next_y_data_new
      
        # restore session if wanted
        if params['resume_training'] == True and i == 0:
            print('... Loading previously trained model from -> ' + save_dir)
            saver.restore(session, save_dir)

        # compute the ramp value
        rmp = 0.0
        if params['ramp'] == True:
            if i>ramp_start:
                rmp = (np.log10(float(i)) - np.log10(ramp_start))/(np.log10(ramp_end) - np.log10(ramp_start))
            if i>ramp_end:
                rmp = 1.0  
        else:
            rmp = 1.0              

        # train the network
        session.run(minimize, feed_dict={bs_ph:bs, x_ph:next_x_data, y_ph:next_y_data, ramp:rmp}) 

        # if we are in a report iteration extract cost function values
        if i % params['report_interval'] == 0 and i > 0:

            # get training loss
            cost, kl, AB_batch = session.run([cost_R, KL, r1_weight], feed_dict={bs_ph:bs, x_ph:next_x_data, y_ph:next_y_data, ramp:rmp})

            # Convert validation x to Hour angle
            x_validation = np.zeros(x_data_test.shape)
            # Get unnormalized array with source parameter truths
            for q_idx,q in enumerate(params['inf_pars']):
                par_min = q + '_min'
                par_max = q + '_max'

                x_validation[:,q_idx] = (x_data_test[:, q_idx] * (bounds[par_max] - bounds[par_min])) + bounds[par_min]
            # convert validation data to hour angle if need be
            x_validation = convert_ra_to_hour_angle(x_validation, params, rand_pars=False)
            # normalize truths again
            for q_idx,q in enumerate(params['inf_pars']):
                par_min = q + '_min'
                par_max = q + '_max'

                x_validation[:,q_idx] = (x_validation[:,q_idx] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])

            # Get validation cost
            cost_val, kl_val = session.run([cost_R, KL], feed_dict={bs_ph:y_data_test.shape[0], x_ph:x_validation, y_ph:y_data_test/y_normscale, ramp:rmp})
            plotdata.append([cost,kl,cost+kl,cost_val,kl_val,cost_val+kl_val])

           
            try:
                # Make loss plot
                plt.figure()
                xvec = params['report_interval']*np.arange(np.array(plotdata).shape[0])
                plt.semilogx(xvec,np.array(plotdata)[:,0],label='recon',color='blue',alpha=0.5)
                plt.semilogx(xvec,np.array(plotdata)[:,1],label='KL',color='orange',alpha=0.5)
                plt.semilogx(xvec,np.array(plotdata)[:,2],label='total',color='green',alpha=0.5)
                plt.semilogx(xvec,np.array(plotdata)[:,3],label='recon_val',color='blue',linestyle='dotted')
                plt.semilogx(xvec,np.array(plotdata)[:,4],label='KL_val',color='orange',linestyle='dotted')
                plt.semilogx(xvec,np.array(plotdata)[:,5],label='total_val',color='green',linestyle='dotted')
                plt.ylim([-25,15])
                plt.xlabel('iteration')
                plt.ylabel('cost')
                plt.legend()
                plt.savefig('%s/latest_%s/cost_%s.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                print('... Saving unzoomed cost to -> %s/latest_%s/cost_%s.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                plt.ylim([np.min(np.array(plotdata)[-int(0.9*np.array(plotdata).shape[0]):,0]), np.max(np.array(plotdata)[-int(0.9*np.array(plotdata).shape[0]):,1])])
                plt.savefig('%s/latest_%s/cost_zoom_%s.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                print('... Saving zoomed cost to -> %s/latest_%s/cost_zoom_%s.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                print()
                plt.close('all')
                
            except:
                pass

            if params['print_values']==True:
                print('--------------------------------------------------------------')
                print('Iteration:',i)
                print('Training -ELBO:',cost)
                print('Validation -ELBO:',cost_val)
                print('Training KL Divergence:',kl)
                print('Validation KL Divergence:',kl_val)
                print('Training Total cost:',kl + cost) 
                print('Validation Total cost:',kl_val + cost_val)
                print()

                # terminate training if vanishing gradient
                if np.isnan(kl+cost) == True or np.isnan(kl_val+cost_val) == True or kl+cost > int(1e5):
                    print()
                    print('Network is returning NaN values')
                    print('Terminating network training')
                    print()
                    if params['hyperparam_optim'] == True:
#                        save_path = saver.save(session,save_dir)
                        nan_flag = True # Declare that nans have been returned use last saved model.
                        return plotdata[-2,5], session, saver, save_dir, nan_flag, plotdata
                    else:
                        exit()
                try:
                    # Save loss plot data
                    np.savetxt(save_dir.split('/')[0] + '/loss_data.txt', np.array(plotdata))
                except FileNotFoundError as err:
                    pass

        # Save model every save interval
        if i % params['save_interval'] == 0 and i > 0:

#            if params['hyperparam_optim'] == False:
                # Save model 
            save_path = saver.save(session,save_dir)
#            else:
#                pass


        # stop hyperparam optim training it and return KL divergence as figure of merit
        if params['hyperparam_optim'] == True and i == params['hyperparam_optim_stop']:
            save_path = saver.save(session,save_dir)
            nan_flag = False

            return np.array(plotdata)[-1,5], session, saver, save_dir, nan_flag, plotdata

        if i % params['plot_interval'] == 0 and i>0:

            n_mode_weight_copy = 100 # must be a multiple of 50
            # just run the network on the test data
            for j in range(params['r']):

                # The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
                if params['n_filters_r1'] != None:
                    XS, dt, _  = run(params, y_data_test[j].reshape([1,y_data_test.shape[1],y_data_test.shape[2]]), np.shape(x_data_test)[1],
                                                 y_normscale, 
                                                 save_dir)
                else:
                    XS, dt, _  = run(params, y_data_test[j].reshape([1,-1]), np.shape(x_data_test)[1],
                                                 y_normscale, 
                                                 save_dir)
                print()
                print('... Runtime to generate {} samples = {} sec'.format(params['n_samples'],dt))            
                print()

                # convert back to RA for plotting
                # get geocenttime index
                for inf_par_idx,inf_par in enumerate(params['inf_pars']):
                    if inf_par == 'ra':
                        ra_idx = inf_par_idx

                # unnormalize and get gps time
                true_post = np.copy(posterior_truth_test[j])
                for inf_par_idx,inf_par in enumerate(params['inf_pars']):
                    XS[:,inf_par_idx] = (XS[:,inf_par_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par+'_min']
                    true_post[:,inf_par_idx] = (true_post[:,inf_par_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par + '_min']

                # convert to RA
                XS = convert_ra_to_hour_angle(XS,params,to_ra=True)
    #                true_post_ra_test[:,ra_idx] -= np.pi

                    # normalize
    #                XS[:,ra_idx]=(XS[:,ra_idx] - bounds['ra_min']) / (bounds['ra_max'] - bounds['ra_min'])
    #                true_post_ra_test[:,ra_idx]=(true_post_ra_test[:,ra_idx] - bounds['ra_min']) / (bounds['ra_max'] - bounds['ra_min'])

                # Make corner plots
                # Get corner parnames to use in plotting labels
                parnames = []
                for k_idx,k in enumerate(params['rand_pars']):
                    if np.isin(k, params['inf_pars']):
                        parnames.append(params['corner_labels'][k])

                defaults_kwargs = dict(
                    bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
                    title_kwargs=dict(fontsize=16),
                    truth_color='tab:orange', quantiles=[0.16, 0.84],
                    levels=(0.68,0.90,0.95), density=True,
                    plot_density=False, plot_datapoints=True,
                    max_n_ticks=3)

                
                if params['pe_dir']==None:
                    figure = corner.corner(XS,**defaults_kwargs,labels=parnames,
                           color='tab:red',
                           fill_contours=True, truths=x_data_test[j,:],
                           show_titles=True)
                else:
                    figure = corner.corner(true_post, **defaults_kwargs,labels=parnames,
                           color='tab:blue',
                           show_titles=True)
#                    figure = corner.corner(posterior_truth_test[j], **defaults_kwargs,labels=parnames,
#                           color='tab:blue',
#                           show_titles=True)
                    corner.corner(XS,**defaults_kwargs,labels=parnames,
                           color='tab:red',
                           fill_contours=True, truths=x_data_test[j,:],
                           show_titles=True, fig=figure)


                plt.savefig('%s/corner_plot_%s_%d-%d.png' % (params['plot_dir'],params['run_label'],i,j))
                print()
                print('... Saved corner plot iteration %d to -> %s/corner_plot_%s_%d-%d.png' % (i,params['plot_dir'],params['run_label'],i,j))
                plt.savefig('%s/latest_%s/corner_plot_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                print('... Saved latest corner plot to -> %s/latest_%s/corner_plot_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                plt.close('all')
                print('... Made corner plot %d' % j)
                print()

    return            

def monte(params, y_data_test, siz_x_data, y_normscale, load_dir, norm_sample,z_batch):
    """ Function to output likelihoods of samples fed in as 
    
    Parameters
    ----------
    params: dict
        Dictionary containing parameter values of the run
    y_data_test: array_like
        test sample time series
    siz_x_data: float
        Number of source parameters to infer
    y_normscale: float
        arbitrary normalization factor for time series
    load_dir: str
        location of pre-trained model
    norm_sample: array_like
        The n original samples from gen_samples run
    nj: int
        Number of time we sample from r1 dist (ie the batch size of this function)
    Returns
    -------
    xs: array_like
        predicted posterior samples
    (run_endt-run_startt): float
        total time to make predicted posterior samples
    mode_weights: array_like
        learned Gaussian Mixture Model modal weights
    """
    multi_modal = True

    # USEFUL SIZES
    xsh1 = siz_x_data #read in from function input variable from [inf params] value in json
    if params['by_channel'] == True:
        ysh0 = np.shape(y_data_test)[0]
        ysh1 = np.shape(y_data_test)[1]
    else:
        ysh0 = np.shape(y_data_test)[1]
        ysh1 = np.shape(y_data_test)[2]
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
            num_det = np.shape(y_data_test)[2]
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
        
        '''
        Set up zj samples
        '''

        zj_samp=r1_dist.sample() # create this to set up the compatible shape with current single sample reconstruction function

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
        masses_loglike = mass_dist.log_prob((tf.boolean_mask(samp_ph,m1_mask,axis=1),tf.boolean_mask(samp_ph,m2_mask,axis=1))) # this could be done in batch if recon allowed
        # dont need to expand dims as masses already output shape (batch,1)
        # print(f'masses loglike = {masses_loglike}')

        '''
        TRUNCATED GAUSSIANS
        '''

        temp_var_r2_gauss = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_gauss)
        gauss_x = tf.boolean_mask(samp_ph,gauss_mask,axis=1)
        tn = tfd.TruncatedNormal(r2_xzy_mean_gauss,tf.sqrt(temp_var_r2_gauss),0.0,1.0)   # shrink the truncation with the ramp
        trunc_gauss_loglike = tf.reduce_sum(tn.log_prob(gauss_x),axis=1)
        trunc_gauss_loglike = tf.expand_dims(trunc_gauss_loglike, axis=1) # get into shape (batch,1)
        # print(f'gauss loglike = {trunc_gauss_loglike}')
        
        '''
        VON MISES (PSI) PART
        '''
        if vonmise_len>0:
            temp_var_r2_vonmise = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_vonmise)
            # print(temp_var_r2_vonmise.shape)
            con_vonmise = tf.reshape(tf.math.reciprocal(temp_var_r2_vonmise),[bs_ph,vonmise_len])   # modelling wrapped scale output as log variance - convert to concentration
            von_mises = tfp.distributions.VonMises(
                          loc=2.0*np.pi*tf.reshape(r2_xzy_mean_vonmise,[bs_ph,vonmise_len]),   # remap 0>1 mean onto 0->2pi range
                          concentration=con_vonmise)
            psi_loglike = von_mises.log_prob(tf.boolean_mask(samp_ph,vonmise_mask,axis=1)) # hopefully the raw norm samp doesnt need remapped like it doews for the training
        
        # tf.reduce_sum(von_mises.log_prob(2.0*np.pi*tf.reshape(tf.boolean_mask(x_ph,vonmise_mask,axis=1),[-1,vonmise_len])),axis=1) # the only thing this is doing is multiplying by 2pi, need to work out if i need to un-engineer this tomorrow





        # '''
        # SKY PARAMS
        # '''
        # # COST FROM RECONSTRUCTION - Von Mises Fisher (sky) parts
        # temp_var_r2_sky = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_sky)
        # con = tf.reshape(tf.math.reciprocal(temp_var_r2_sky),[bs_ph])   # modelling wrapped scale output as log variance - only 1 concentration parameter for all sky
        # loc_xyz = tf.math.l2_normalize(tf.reshape(r2_xzy_mean_sky,[-1,3]),axis=1)    # take the 3 output mean params from r2 and normalse so they are a unit vector
        # von_mises_fisher = tfp.distributions.VonMisesFisher(
        #               mean_direction=loc_xyz,
        #               concentration=con)
        
        
        # ra_sky = 2*np.pi*tf.reshape(tf.boolean_mask(samp_ph,ra_mask,axis=1),[-1,1])       # convert the scaled 0->1 true RA value back to radians
        # dec_sky = np.pi*(tf.reshape(tf.boolean_mask(samp_ph,dec_mask,axis=1),[-1,1]) - 0.5) # convert the scaled 0>1 true dec value back to radians
        # xyz_unit = tf.reshape(tf.concat([tf.cos(ra_sky)*tf.cos(dec_sky),tf.sin(ra_sky)*tf.cos(dec_sky),tf.sin(dec_sky)],axis=1),[-1,3])   # construct the true parameter unit vector
        # sky_loglike = von_mises_fisher.log_prob(tf.math.l2_normalize(xyz_unit,axis=1))   # normalise it for safety (should already be normalised) and compute the logprob
        # sky_loglike = tf.expand_dims(sky_loglike, axis=1)

        # print(f'sky loglike = {sky_loglike.get_shape()}')


        '''
        COMBINE LOGLIKES
        TODO check if these 2 methods underneat hgive the same result...DONE - they both give the same result, ask Chris is one is preferable:
        '''
        # single_loglike_cheap=tf.squeeze(tf.reduce_sum([masses_loglike + trunc_gauss_loglike + sky_loglike],axis=0)) # method 1
        single_loglike=tf.concat([masses_loglike,trunc_gauss_loglike,psi_loglike],axis=1) # method 2 (2 lines) get into shape (batch,3)
        single_loglike=tf.reduce_sum(single_loglike, axis=1) # then reduce sum over the axis=1 to get shape (batch) (1d array)




        # print(f'single loglike size = {single_loglike.get_shape()}')
        # loglike_ph=tf.concat([loglike_ph, [single_loglike]],axis=0)
        
        # loglike_ph=tf.slice(loglike_ph, [1], [Nj]) # chop off the first [0] placeholder entry to leave only the summed loglikes of the 3 dists, one entry fror each Nj iteration
        
        '''
        CALC EXPECTATION VALUE (2 METHODS)
        '''

        final_loglike=tf.reduce_logsumexp(single_loglike, axis=0) # calc expectation value method 1



        # print(f'final loglike size = {final_loglike.get_shape()}')
        # final_loglike=tf.subtract(tf.reduce_logsumexp(loglike_ph, axis=0), tf.log(tf.Variable(,dtype=tf.dtypes.float32))) # calc expectation value method 2

        # print(f'norm samples = {norm_sample[0,...].shape}')

        '''
        Run Session
        '''
        init = tf.initialize_all_variables()
        session.run(init)
        saver_VICI = tf.train.Saver(tf.global_variables())
        saver_VICI.restore(session,load_dir)
    # ns = z_batch # number of zj samps done in parallel = batch size 
    y_data_test_exp = np.tile(y_data_test,(z_batch,1))/y_normscale
    y_data_test_exp = y_data_test_exp.reshape(-1,params['ndata'],num_det)

    norm_sample_tiled = np.tile(norm_sample,(z_batch,1))
    single_graph_1=time.time()
    loglike, sky_log, single_log, trunc_log, mass_log = session.run([final_loglike, sky_loglike, single_loglike, trunc_gauss_loglike, masses_loglike],feed_dict={samp_ph: norm_sample_tiled, bs_ph: z_batch, y_ph: y_data_test_exp})
    single_graph_2=time.time()

    single_graph_time=single_graph_2-single_graph_1

    # print(f'time to gen {ns} z batch sampels and likelihood={t2-t1}')

    # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    # # print(f'z samples = {ns}')
    # print(f'mass shape = {mass_log.shape}')
    # print(f'trunc shape = {trunc_log.shape}')
    # print(f'sky shape = {sky_log.shape}')
    # # print(trunc_log_full)
    # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    # print(trunc_log)

    # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    # print(f'single shape = {single_log.shape}')
    # print(f'final = {loglike} with shape {loglike.shape}')

    return loglike,single_graph_time


'''
########################################################################################################
Here is where I will experiment with the bilby stuff
########################################################################################################
'''

########################################################################################################


# def bilby_stuff(fixed_vals, params, bounds,
#                         #  x_data_test, # this is array like, but might need to make it a dict without other test_set readin
#                         vit_loglikes, vit_samples, # need y data, FT to freq dom
#                         uufd,

#                         # seed=None, outdir='./importance_sampling_results', start_sample=0, end_sample=5000,
#                         ):

#     '''
#     INPUTS:
#     - fixed_vals, params, bounds all come from the params.jsons (x3) which are global in the run_vitamin script. 
#     # - x_data_test: changing this from a dictionary to simply a 1d array as the dict is redundant # a dictionary from the h5py file with 9 rand params with their final values
#     - vit loglikes: 1d array of loglikes of vitamin samples from monte function
#     - vit_samples: 2d array (nsamp,nparam) where len axis=0 is the same as len vit loglikes 
#     - uufd: fd strain data from test_set as isolated in run_vit.gen_samples parent function
#     '''

#     ref_geocent_time = params['ref_geocent_time']
#     duration = params['duration']
#     psd_files = params['psd_files'] # if leave blank = bilby (make it an empty list), note its empty in params json so can leave this
#     sampling_frequency = params['ndata']
#     inf_pars=params['inf_pars']
#     rand_pars=params['rand_pars']


#     # masks for param extraction for likelihood evaluation...

#     gauss_mask, gauss_idx_mask, gauss_len = get_param_index(params['inf_pars'],params['gauss_pars'])
#     ra_mask, ra_idx_mask, ra_len = get_param_index(params['inf_pars'],['ra'])
#     dec_mask, dec_idx_mask, dec_len = get_param_index(params['inf_pars'],['dec'])
#     m1_mask, m1_idx_mask, m1_len = get_param_index(params['inf_pars'],['mass_1'])
#     m2_mask, m2_idx_mask, m2_len = get_param_index(params['inf_pars'],['mass_2'])

#     # define the start time of the timeseries
#     start_time = ref_geocent_time-duration/2.0 # start time to inject signals

#     # choose waveform parameters here, only thing this edits from the input d_test_data variable is the geocent time.
#     # pars = fixed_vals
#     # # print(f'this is pars1 bro: {pars}')
#     # for par_idx, par in enumerate(params['rand_pars']):
#     #     if par == 'geocent_time':
#     #         pars[par] = x_data_test[par_idx] + ref_geocent_time
#     #     else:
#     #         pars[par] = x_data_test[par_idx]

#     # print(f'this is pars2 bro: {pars}')

#     '''
#     CHECKPOINT: we now have 'pars' which is a dict that originated as all 15 fixed vals params from json param. This was then overwritten by the ACTUAL
#     x data from the h5py file for the 9 rand pars. So we have 15 vals, 6 fixed and 9 the true source params.
#     Ask Chris what to do if we want to inject signal to match waveform that we DONT KNOW the true x data for. Hunter suggested use the the values of the
#     vitamin posterior with the highest loglike (MLE approx)
#     '''

#     '''Starting again using gen_real_events as a nice benchmark'''

#     injection_parameters=None

#         # First, put our "data" created above into a list of intererometers (the order is arbitrary)
#     ifos = bilby.gw.detector.InterferometerList(params['det'])
#     for ifo_ind, ifo in enumerate(ifos):
#         ifo.set_strain_data_from_frequency_domain_strain(uufd[ifo_ind,:],
#                                                     sampling_frequency=sampling_frequency,
#                                                     duration=duration,
#                                                     start_time=start_time)

#     prior = bilby.core.prior.PriorDict() # need to give phase prior for phase marginalisation!!!
#     prior['phase'] = bilby.gw.prior.Uniform(name='phase', minimum=bounds['phase_min'], maximum=bounds['phase_max'], boundary='periodic')



#     # # prior['mass_1'] = bilby.gw.prior.Uniform(name='mass_1', minimum=bounds['mass_1_min'], maximum=bounds['mass_1_max'],unit='$M_{\odot}$')
#     # # prior['mass_2'] = bilby.gw.prior.Uniform(name='mass_2', minimum=bounds['mass_2_min'], maximum=bounds['mass_2_max'],unit='$M_{\odot}$')
#     # prior['geocent_time'] = bilby.core.prior.Uniform(
#     #         minimum=ref_geocent_time + bounds['geocent_time_min'],
#     #         maximum=ref_geocent_time + bounds['geocent_time_max'],
#     #         name='geocent_time', latex_label='$t_c$', unit='$s$')
#     # prior['a_1'] =  0.0
#     # prior['a_2'] =  0.0
#     # prior['tilt_1'] =  0.0
#     # prior['tilt_2'] =  0.0
#     # prior['phi_12'] =  0.0
#     # prior['phi_jl'] =  0.0
#     # prior['dec'] =  -1.2232
#     # prior['ra'] =  2.19432
#     # prior['theta_jn'] =  1.89694
#     # prior['psi'] =  0.532268
#     # prior['luminosity_distance'] = 412.066

#     # Next create a dictionary of arguments which we pass into the LALSimulation waveform - we specify the waveform approximant here
#     waveform_arguments = dict(waveform_approximant='IMRPhenomPv2', # NR model selection, the same one as vit traind on
#                               reference_frequency=20., minimum_frequency=20.)

#     # Next, create a waveform_generator object. This wraps up some of the jobs of converting between parameters etc
#     waveform_generator = bilby.gw.WaveformGenerator(
#         frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
#         waveform_arguments=waveform_arguments,
#         parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters)

#     # Finally, create our likelihood, passing in what is needed to get going
#     likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
#         ifos, waveform_generator, 
#         priors=prior,
#         time_marginalization=False, phase_marginalization=True, distance_marginalization=False)

    

#     # print(np.ma.masked_array(vit_samples[0], mask=m1_mask*1))

#     # print(vit_samples[0][0])

    
    
#     # weights = []

#     # print(f'vit loglikes {vit_loglikes}')

#     # for i in range(number_of_samples): # might need to give it all of them but the ones we dont infer might come from the fixed vals.

#     '''
#     PSEUDOCODE
#     using tensorflow here and utlising the batch functionality to do monte carlo integration (sum then divide)
#     the difficulty is it's log space so need to do some type of logsumexp
#     Everything up to this point in this bilby function has been preamble and thus the same for any sample or any psi int loop.
#     Thus, there is little scope for speeding up by tf. I read in the uufd only once,
#     I also need to look at the possibility of keeping the batch dim for nsamp instead but will talk to crhis about this.
#     I assume i need to np.tile at some point but we shall see. 
#     step 1 - create a tensorflow graph
#     step 2 - create uniform dist between 0 and pi
#     step 3 - sample from this dist N times (in batch at once)
#     step 4 - assign N sets of likelihood parameters in batch
#     step 5 - this gives a batch of N loglikes
#     step 6 - log sum exp across the N-length batch dimension
#     step 7 - divide through by N
    
#     For clarity, Im going to do it without tensorflow and see how long it takes.
#     goal is to get it in batch using tf in the daughter function then loop the parent function like in the vit loglikes
#     ask chris his recomendation for getting this bilby bit into a single tf batch
#     '''

#     def progress(count, total, suffix=''):
#         bar_len = 60
#         filled_len = int(round(bar_len * count / float(total)))

#         percents = round(100.0 * count / float(total), 1)
#         bar = '=' * filled_len + '-' * (bar_len - filled_len)

#         sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
#         sys.stdout.flush()  # As suggested by Rom Ruben

#     # set up full array framework:
#     N_psi = 1000 # how many times to sample for psi marginalisation
#     number_of_samples = len(vit_loglikes)
#     bilby_loglikes=np.zeros([number_of_samples,N_psi])

#     print('########################################################################################')
#     print(f'#### BILBY LOGLIKELIHOODS: number of vitamin samples looped: {number_of_samples}. psi_batchsize = {N_psi} ####')
#     print('########################################################################################')    



#     for i in range(number_of_samples):
#         progress(i+1,number_of_samples,'')#f'           Calculating Bilby Likelihoods for {number_of_samples} VItamin Samples')# f'Calculating Loglikelihood for {num_samples} VItamin sample(s)')
#         psi_samples = np.random.uniform(0,np.pi,N_psi) # overwrites itself each vit_sample 

#         for ind, sample in enumerate(psi_samples):
#             # print(sample) # check it worked and it does!
#             likelihood_parameters = dict(

#                 # fixed for one samples
#                 mass_1=vit_samples[i,...][0], # might want to use placeholders inside tf graph, we'll see
#                 mass_2=vit_samples[i,...][1],
#                 luminosity_distance=vit_samples[i,...][2],
#                 geocent_time=vit_samples[i,...][3],
#                 theta_jn=vit_samples[i,...][4], 
#                 ra=vit_samples[i,...][5], # option to simplify is to get rid of ra and dec. 
#                 dec=vit_samples[i,...][6], # not flat prior, sinusoid prior (try convert to a space that emulates flat prior somehow)
#                 phase=0, # can set to any float and it doesn't change the overall value due to phase marginalisation.
#                 a_1=fixed_vals['a_1'], a_2=fixed_vals['a_2'], tilt_1=fixed_vals['tilt_1'], tilt_2=fixed_vals['tilt_2'], phi_12=fixed_vals['phi_12'], phi_jl=fixed_vals['phi_jl'], # all 6 of these vals are zero

#                 # changes N_psi times per sample
#                 psi=sample,
#                 )

#             '''
#             scatter plot of pairs of params where color is bilby likelihood
#             '''

#             # grid approach for psi is more optimum than random sampling

#             likelihood.parameters = likelihood_parameters
#             bilby_loglike_single = likelihood.log_likelihood()
#             bilby_loglikes[i,ind]=bilby_loglike_single

#     # print(bilby_loglikes.shape)

#     bilby_loglike_means=logsumexp(bilby_loglikes,axis=1) # need to set axis i think

#     # print(bilby_loglike_means.shape)


        
#         # mass_1=75,
#         # mass_2=40,
#         # luminosity_distance=fixed_vals['luminosity_distance'],
#         # geocent_time=fixed_vals['geocent_time'],
#         # theta_jn=fixed_vals['theta_jn'], 
#         # ra=fixed_vals['ra'], 
#         # dec=fixed_vals['dec'],

#         #rand_pars
#         # psi=1, # need to scipy quad this over all psi vals, uniform dist between 0 and pi. just set to 1 now for ease.
#         # phase=posterior_dict_old['phase'][i],
#         #  need to average over like, not loglike, need to logsumexp.

#         # '''
#         # Do i need to feed in the other 6 fixed vals too to get all 15 likelihood pars or nah?
#         # '''
#         # note, all 6 of this spin params are fixed at zero

        

#     # print(likelihood_parameters['mass_1'])

#     # bilby_loglike = []
#     # likelihood.parameters = likelihood_parameters
#     # bilby_loglike = likelihood.log_likelihood() # dont want ratio look up. got rid of ratio

#     # print(f'bilby_loglikes are {bilby_loglike_means}')

#     '''IS starts
#     TODO - plot different likes
#     TODO - start IS reweighting
#     TODO - speed up a lot!!!
#     ''' 

#     # weight = np.exp(bilby_loglike_single - vitamin_loglikes[i]) # instead of this, do it as 2 arrays with np.subtract to get full array of weights instead of element-wise

#     # bilby_loglikes.append(bilby_loglike_single)
#     # weights.append(weight)

#     return bilby_loglike_means



#     # fix parameters here not the params not in rand_pars then default to their fixed value by nature of the pars dict i just created
#     # injection_parameters = dict(
#     #     mass_1=pars['mass_1'],mass_2=pars['mass_2'],
#     #     a_1=pars['a_1'], a_2=pars['a_2'], tilt_1=pars['tilt_1'], tilt_2=pars['tilt_2'],
#     #     phi_12=pars['phi_12'], phi_jl=pars['phi_jl'], 
#     #     luminosity_distance=pars['luminosity_distance'], theta_jn=pars['theta_jn'], psi=pars['psi'],
#     #     phase=pars['phase'], geocent_time=pars['geocent_time'], ra=pars['ra'], dec=pars['dec'])

#     # Fixed arguments passed into the source model
#     waveform_arguments = dict(waveform_approximant='IMRPhenomPv2', # NR model selection, the same one as vit traind on
#                               reference_frequency=20., minimum_frequency=20.)

#     # Create the waveform_generator using a LAL BinaryBlackHole source function
#     waveform_generator = bilby.gw.WaveformGenerator(
#         duration=params['duration'], sampling_frequency=params['ndata'],
#         frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
#         parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
#         waveform_arguments=waveform_arguments,
#         start_time=start_time)

#     # create waveform
#     wfg = waveform_generator

#     # extract waveform from bilby
#     wfg.parameters = injection_parameters
#     freq_signal = wfg.frequency_domain_strain() # gets noisy freq domain sig
#     time_signal = wfg.time_domain_strain() # gets noisy time domain sig- gets from bilby

#     # Set up interferometers. These default to their design

#     ''' if get weird results try new o4 model on github docs *NEW* and test waveforms too'''

#     # sensitivity
#     ifos = bilby.gw.detector.InterferometerList(params['det']) # can get rid of as repeats line above

#     # set noise to be colored Gaussian noise (can ignore, juyst for bilby admin)
#     ifos.set_strain_data_from_power_spectral_densities(
#     sampling_frequency=params['ndata'], duration=duration,
#     start_time=start_time)

#     # inject signal into colored gauss noise
#     ifos.inject_signal(waveform_generator=waveform_generator,
#                        parameters=injection_parameters)


#     '''
#     #### PRIORS ####

#     Now time to create priors. Im really unsure about this cause Hunter's original code had dict.pop('chirp_mass') which makes sense now that I think about it cause
#     its the only one in dict(pars) that is a None value so let's keep this.

#     The problem with this code that it only sets non-fixed priors for inf_pars BUT this doesn't include phase and phi. But I want a phase marginalised likelihood
#     from bilby I need the phase prior to not be fixed so there is a conflict there.

#     Want a prior for each injection par (but i'm popping out chirp mass as it's NONE in pars)

#     '''



#     mypriors = bilby.gw.prior.BBHPriorDict()
#     # i pop the ones i dont want?
#     mypriors.pop('chirp_mass')

#     # priors['mass_ratio'] = bilby.gw.prior.Constraint(minimum=0.125, maximum=1, name='mass_ratio', latex_label='$q$', unit=None)
    
#     # inf pars...

#     if np.any([r=='mass_1' for r in rand_pars]):
#         mypriors['mass_1'] = bilby.gw.prior.Uniform(name='mass_1', minimum=bounds['mass_1_min'], maximum=bounds['mass_1_max'],unit='$M_{\odot}$')
#     else:
#         mypriors['mass_1'] = fixed_vals['mass_1']

#     if np.any([r=='mass_2' for r in rand_pars]):
#         mypriors['mass_2'] = bilby.gw.prior.Uniform(name='mass_2', minimum=bounds['mass_2_min'], maximum=bounds['mass_2_max'],unit='$M_{\odot}$')
#     else:
#         mypriors['mass_2'] = fixed_vals['mass_2']

#     if np.any([r=='luminosity_distance' for r in rand_pars]):
#         mypriors['luminosity_distance'] =  bilby.gw.prior.Uniform(name='luminosity_distance', minimum=bounds['luminosity_distance_min'], maximum=bounds['luminosity_distance_max'], unit='Mpc')
#     else:
#         mypriors['luminosity_distance'] = fixed_vals['luminosity_distance']

#     if np.any([r=='geocent_time' for r in rand_pars]): # need to read in inf pars = params['inf_pars']
#         mypriors['geocent_time'] = bilby.core.prior.Uniform(
#             minimum=ref_geocent_time + bounds['geocent_time_min'],
#             maximum=ref_geocent_time + bounds['geocent_time_max'],
#             name='geocent_time', latex_label='$t_c$', unit='$s$')
#     else:
#         mypriors['geocent_time'] = fixed_vals['geocent_time']

#     if np.any([r=='theta_jn' for r in rand_pars]):
#         pass
#     else:
#         mypriors['theta_jn'] = fixed_vals['theta_jn']

#     if np.any([r=='ra' for r in rand_pars]):
#         mypriors['ra'] = bilby.gw.prior.Uniform(name='ra', minimum=bounds['ra_min'], maximum=bounds['ra_max'], boundary='periodic')
#     else:
#         mypriors['ra'] = fixed_vals['ra']

#     if np.any([r=='dec' for r in rand_pars]):
#         pass
#     else:    
#         mypriors['dec'] = fixed_vals['dec']

#     # rand pars...

#     if np.any([r=='phase' for r in rand_pars]): # marginalising over this
#         mypriors['phase'] = bilby.gw.prior.Uniform(name='phase', minimum=bounds['phase_min'], maximum=bounds['phase_max'], boundary='periodic')
#     else:
#         mypriors['phase'] = fixed_vals['phase']

#     if np.any([r=='psi' for r in rand_pars]): # need to int to manually marginalise over this
#         mypriors['psi'] = bilby.gw.prior.Uniform(name='psi', minimum=bounds['psi_min'], maximum=bounds['psi_max'], boundary='periodic')
#     else:
#         mypriors['psi'] = fixed_vals['psi']

#     # other injection pars...

#     if np.any([r=='a_1' for r in rand_pars]):
#         mypriors['a_1'] = bilby.gw.prior.Uniform(name='a_1', minimum=bounds['a_1_min'], maximum=bounds['a_1_max'])
#     else:
#         mypriors['a_1'] = fixed_vals['a_1']

#     if np.any([r=='a_2' for r in rand_pars]):
#         mypriors['a_2'] = bilby.gw.prior.Uniform(name='a_2', minimum=bounds['a_2_min'], maximum=bounds['a_2_max'])
#     else:
#         mypriors['a_2'] = fixed_vals['a_2']

#     if np.any([r=='tilt_1' for r in rand_pars]):
#         #   mypriors['tilt_1'] = bilby.gw.prior.Uniform(name='tilt_1', minimum=bounds['tilt_1_min'], maximum=bounds['tilt_1_max'])
#         pass
#     else:
#         mypriors['tilt_1'] = fixed_vals['tilt_1']

#     if np.any([r=='tilt_2' for r in rand_pars]):
# #           mypriors['tilt_2'] = bilby.gw.prior.Uniform(name='tilt_2', minimum=bounds['tilt_2_min'], maximum=bounds['tilt_2_max'])
#         pass
#     else:
#         mypriors['tilt_2'] = fixed_vals['tilt_2']

#     if np.any([r=='phi_12' for r in rand_pars]):
#         mypriors['phi_12'] = bilby.gw.prior.Uniform(name='phi_12', minimum=bounds['phi_12_min'], maximum=bounds['phi_12_max'], boundary='periodic')
#     else:
#         mypriors['phi_12'] = fixed_vals['phi_12']

#     if np.any([r=='phi_jl' for r in rand_pars]):
#         mypriors['phi_jl'] = bilby.gw.prior.Uniform(name='phi_jl', minimum=bounds['phi_jl_min'], maximum=bounds['phi_jl_max'], boundary='periodic')
#     else:
#         mypriors['phi_jl'] = fixed_vals['phi_jl']
    
#     # print(mypriors)



    # # Create the GW likelihood
    # bilby_likelihood_template = bilby.gw.likelihood.GravitationalWaveTransient(
    #     interferometers=ifos, waveform_generator=waveform_generator,
    #     time_marginalization=False, phase_marginalization=True, distance_marginalization=False,
    #     priors=mypriors,
    #     )

    
    # vitamin_loglikes = vit_loglikes # 1d array of loglikes from my monte
    # number_of_samples = len(vit_loglikes)


    # bilby_loglikes = []
    # weights = []

    # print(f'vit loglikes {vit_loglikes}')

    # for i in range(number_of_samples): # might need to give it all of them but the ones we dont infer might come from the fixed vals.

    #     likelihood_parameters = dict(

    #         # inf_pars
    #         mass_1=tf.boolean_mask(vitamin_samples[i,...],m1_mask),
    #         mass_2=tf.boolean_mask(vitamin_samples[i,...],m2_mask),
    #         luminosity_distance=tf.boolean_mask(vitamin_samples[i,...],gauss_mask)[0],
    #         geocent_time=tf.boolean_mask(vitamin_samples[i,...],gauss_mask)[1],
    #         theta_jn=tf.boolean_mask(vitamin_samples[i,...],gauss_mask)[2], 
    #         ra=tf.boolean_mask(vitamin_samples[i,...],ra_mask), 
    #         dec=tf.boolean_mask(vitamin_samples[i,...],dec_mask),
            
    #         #rand_pars
    #         # psi=1, # need to scipy quad this over all psi vals, uniform dist between 0 and pi. just set to 1 now for ease.
    #         # phase=posterior_dict_old['phase'][i],
    #         psi=0, # need to average over like, not loglike, need to logsumexp.
    #         # phase=pars['phase'],

    #         # '''
    #         # Do i need to feed in the other 6 fixed vals too to get all 15 likelihood pars or nah?
    #         # '''
    #         a_1=fixed_vals['a_1'], a_2=fixed_vals['a_2'], tilt_1=fixed_vals['tilt_1'], tilt_2=fixed_vals['tilt_2'], phi_12=fixed_vals['phi_12'], phi_jl=fixed_vals['phi_jl'], 
    #         )



    #     '''IS starts'''

    #     bilby_likelihood_template.parameters = likelihood_parameters
    #     bilby_loglike_single = bilby_likelihood_template.log_likelihood_ratio() # dont want ratio look up.

    #     weight = np.exp(bilby_loglike_single - vitamin_loglikes[i])

    #     bilby_loglikes.append(bilby_loglike_single)
    #     weights.append(weight)

    # return bilby_loglikes