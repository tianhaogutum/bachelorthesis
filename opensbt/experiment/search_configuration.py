# DKM - file changed from original

class SearchConfiguration(object):
    n_generations = None 
    population_size = None
    maximal_execution_time = None
    num_offsprings = None
    prob_crossover = None
    eta_crossover = None
    prob_mutation = None
    eta_mutation = None
    set_seed = None
    
    # NSGAII-DT specific
    max_tree_iterations = None
    inner_num_gen = None
    n_func_evals_lim = None

    # metrics
    ref_point_hv = None
    nadir = None
    ideal = None
    

#TODO create a search configuration file specific for each algorithm
class DefaultSearchConfiguration(SearchConfiguration):
    n_generations = 5 
    population_size = 20
    maximal_execution_time = None
    num_offsprings = None
    prob_crossover = 0.7
    eta_crossover = 20
    prob_mutation = None
    eta_mutation = 15
    set_seed = 1

    # NSGAII-DT specific
    inner_num_gen = 4
    max_tree_iterations = 4
    n_func_evals_lim = 500

    # metrics
    ref_point_hv = None
    nadir = ref_point_hv
