import bilby

# Raw json loglikes

bilby_json = '/scratch/wiay/matthewd/msci_project/vitamin_b/vitamin_b/test_sets/all_4_samplers/test_dynesty1/all_4_samplers_0_result.json'
result_file = bilby.result.read_in_result(filename=bilby_json)
bilby_raw_loglikes = result_file.log_likelihood_evaluations

# uufd

