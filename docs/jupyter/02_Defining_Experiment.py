# Script version of 02_Defining_Experiment.ipynb

import pymoo
import os

# Define the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

# Home directory is two up from current directory
home_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
os.chdir(home_dir)
print(os.getcwd())

# DM - added prefix "opensbt." to imports, assuming home directory is opensbt-core
from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.experiment.search_configuration import DefaultSearchConfiguration

from opensbt.evaluation.fitness import *
from opensbt.problem.adas_problem import ADASProblem
from opensbt.problem.pymoo_test_problem import PymooTestProblem
from opensbt.experiment.experiment_store import *
from opensbt.algorithm.algorithm import *
from opensbt.evaluation.critical import *
from opensbt.simulation.dummy_simulation import DummySimulator

from opensbt.utils import log_utils

# Define Problem
problem = ADASProblem(
                      problem_name="DummySimulatorProblem",
                      scenario_path="scenarios/dummy_scenario.xosc",
                      xl=[0, 1, 0, 0.2],
                      xu=[180, 10,180, 3],
                      simulation_variables=[
                          "orientation_ego",
                          "velocity_ego",
                          "orientation_ped",
                          "velocity_ped"],
                      fitness_function=FitnessMinDistanceVelocity(),
                      critical_function=CriticalAdasDistanceVelocity(),
                      simulate_function=DummySimulator.simulate,
                      simulation_time=10,
                      sampling_time=0.25
                      )

# Run the experiment
log_utils.setup_logging("./log.txt")

# Set search configuration
config = DefaultSearchConfiguration()
config.n_generations = 50
config.population_size = 20

# Instantiate search algorithm
optimizer = NsgaIIOptimizer(
                            problem=problem,
                            config= config)

# Run search
res = optimizer.run()

# Write results
res.write_results(params = optimizer.parameters)
