import os
import pymoo
import numpy as np
from typing import Tuple

# set up pymoo logging
from opensbt.mocddel_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

# import opensbt
from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from opensbt.evaluation.fitness import *
from opensbt.problem.adas_problem import ADASProblem
from opensbt.problem.pymoo_test_problem import PymooTestProblem
from opensbt.experiment.experiment_store import *
from opensbt.algorithm.algorithm import *
from opensbt.evaluation.critical import *
from opensbt.simulation.dummy_simulation import DummySimulator

# dependencies of fitness function
from opensbt.evaluation.fitness import Fitness
from opensbt.simulation.simulator import SimulationOutput
from opensbt.utils import geometric

# dependencies of critical function
from opensbt.evaluation.critical import Critical
from opensbt.simulation.simulator import SimulationOutput

# import pymoo to apply search algorithm
from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer

from opensbt.utils import log_utils

from opensbt.simulation.dummy_simulation import DummySimulator

# define the fitness function
class MyFitnessFunction(Fitness):
    @property
    def min_or_max(self):
        return "min", "max"

    @property
    def name(self):
        return "Min distance", "Velocity at min distance"

    def eval(self, simout: SimulationOutput) -> Tuple[float]:
        traceEgo = simout.location["ego"]                     # The actor names have to match the names written into the simulationoutput instance by the simulation adapter
        tracePed = simout.location["adversary"]              

        ind_min_dist = np.argmin(geometric.distPair(traceEgo, tracePed))

        # distance between ego and other object
        distance = np.min(geometric.distPair(traceEgo, tracePed))

        # speed of ego at time of the minimal distance
        speed = simout.speed["ego"][ind_min_dist]

        return (distance, speed)
    
# define the critical function
class MyCriticalFunction(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        # Corresponds F1 < 0.5 and F2 > 0 (inverted because minimization is performed)
        if (vector_fitness[0] < 0.5) and (vector_fitness[1] < 0):
            return True
        else:
            return False
        

# define the problem
problem = ADASProblem(
                      problem_name="DummySimulatorProblem",
                      scenario_path="scenarios/dummy_scenario.xosc",
                      xl=[0, 1, 0, 0.5],
                      xu=[180, 10,180, 3],
                      simulation_variables=[
                          "orientation_ego",
                          "velocity_ego",
                          "orientation_ped",
                          "velocity_ped"],
                      fitness_function=MyFitnessFunction(),
                      critical_function=MyCriticalFunction(),
                      simulate_function=DummySimulator.simulate,
                      simulation_time=10,
                      sampling_time=0.25
                      )

log_utils.setup_logging("./log.txt")

# Set search configuration
config = DefaultSearchConfiguration()
config.n_generations = 50
config.population_size = 20

# Instantiate search algorithm
optimizer = NsgaIIOptimizer(
                            problem=problem,
                            config= config)

res = optimizer.run()
res.write_results(params = optimizer.parameters)
