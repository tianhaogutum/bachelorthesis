import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

import numpy as np

from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer

import logging as log
import os

from examples.lanekeeping.donkey.donkey_simulation import DonkeySimulator
from opensbt.config import LOG_FILE

from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from opensbt.problem.adas_problem import ADASProblem
from opensbt.utils.log_utils import disable_pymoo_warnings, setup_logging
from examples.lanekeeping.evaluation import fitness
from examples.lanekeeping.evaluation import critical

os.chmod(os.getcwd(), 0o777)
logger = log.getLogger(__name__)
setup_logging(LOG_FILE)
disable_pymoo_warnings()

problem = ADASProblem(
    problem_name="Donkey_5A_0-85_XTE_AVG",
    scenario_path="",
    xl=[0, 0, 0, 0, 0],
    xu=[85, 85, 85,85, 85],
    simulation_variables=[
    "angle1",
    "angle2",
    "angle3",
    "angle4",
    "angle5"],
    fitness_function=fitness.MaxAvgXTEFitness(),
    critical_function=critical.MaxXTECriticality(),
    simulate_function=DonkeySimulator.simulate,
    simulation_time=30,
    sampling_time=0.25,
)

# Set search configuration
config = DefaultSearchConfiguration()
config.n_generations = 2
config.population_size = 1

config.ideal = np.asarray([-3,-10])    # worst (=most critical) fitness values
config.nadir = np.asarray([0,0])    # worst (=most critical) fitness values

optimizer = NsgaIIOptimizer(
                            problem=problem,
                            config=config)

res = optimizer.run()
res.write_results(params=optimizer.parameters)