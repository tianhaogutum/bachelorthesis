# MIT License
# 
# Copyright (c) [Year] [Your Name]
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from typing import Tuple
import pymoo
import os
import time
import subprocess
import json
import logging
logging.getLogger().setLevel(logging.ERROR)
from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated
from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended
from opensbt.model_ga.result import SimulationResult
pymoo.core.result.Result = SimulationResult
from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem
from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from opensbt.evaluation.fitness import *
from opensbt.experiment.experiment_store import *
from opensbt.algorithm.algorithm import *
from opensbt.evaluation.critical import *
from opensbt.evaluation.fitness import Fitness
from opensbt.utils import geometric
from opensbt.evaluation.critical import Critical
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
from typing import List
from opensbt.simulation.simulator import Simulator, SimulationOutput
import numpy as np
from opensbt.utils.geometric import *
import logging as log
from opensbt.utils import geometric
import json
from dataclasses import dataclass
from pymoo.core.problem import Problem
import numpy as np
from opensbt.evaluation.critical import Critical
from opensbt.evaluation.fitness import *
import logging as log
import matplotlib.pyplot as plt

@dataclass
class SWARMProblem(Problem):
    f_out = {}
    counter = 0

    def __init__(self,
                 xl: List[float],
                 xu: List[float],
                 fitness_function: Fitness,
                 simulate_function,
                 critical_function: Critical,
                 simulation_variables: List[float],
                 simulation_time: float = 10,
                 sampling_time: float = 100,
                 problem_name: str = None,
                 do_visualize: bool = False):

        super().__init__(n_var=len(xl),
                         n_obj=len(fitness_function.name),
                         xl=xl,
                         xu=xu)

        assert xl is not None
        assert xu is not None
        assert fitness_function is not None
        assert simulate_function is not None
        assert simulation_time is not None
        assert sampling_time is not None
        assert np.equal(len(xl), len(xu))
        assert np.less_equal(xl, xu).all()
        assert len(fitness_function.min_or_max) == len(fitness_function.name)

        self.fitness_function = fitness_function
        self.simulate_function = simulate_function
        self.critical_function = critical_function
        self.simulation_time = simulation_time
        self.sampling_time = sampling_time
        self.simulation_variables = simulation_variables
        self.do_visualize = do_visualize
        self.problem_name = problem_name

        self.signs = []
        for value in self.fitness_function.min_or_max:
            if value == 'max':
                self.signs.append(-1)
            elif value == 'min':
                self.signs.append(1)
            else:
                raise ValueError(
                    "Error: The optimization property " + str(value) + " is not supported.")

    def _evaluate(self, x, out, *args, **kwargs):
        log.info(f"Running evaluation number {SWARMProblem.counter}")
        filename = 'stat_results_average_distance_with_obsts.txt'
        with open(filename, 'a') as file:
            file.write(f"simulation variables: {x}\n") 
        try:
            simout_list = self.simulate_function(x, self.simulation_variables)
        except Exception as e:
            log.info("Exception during simulation occurred: ")
            raise e
        out["SO"] = []
        vector_list = []
        label_list = []
        print("The simulation of this generation is done, now step into evaluation")
        for simout in simout_list:
            out["SO"].append(simout)
            vector_fitness = np.asarray(self.signs) * np.array(self.fitness_function.eval(simout))
            vector_list.append(np.array(vector_fitness))
            label_list.append(self.critical_function.eval(vector_fitness))
            print(self.signs)
        out["F"] = np.vstack(vector_list)
        out["CB"] = label_list
        with open(filename, 'a') as file:
           file.write(f"evaluation: {out}\n") 
        SWARMProblem.f_out[SWARMProblem.counter] = out["F"]
        SWARMProblem.counter += 1
            
    def is_simulation(self):
        return True
 
    @staticmethod
    def plot_generation_data(data):
        plt.figure()
        x = []
        y = []
        counts = []
        for generation_num, values in data.items():
            generation_num = int(generation_num)
            values = np.array(values)
            for value in values:
                x.append(generation_num)
                y.append(value)
        
        x = np.array(x)
        y = np.array(y)
        points = np.column_stack((x, y))
        unique_points, counts = np.unique(points, axis=0, return_counts=True)

        for point, count in zip(unique_points, counts):
            plt.scatter(point[0], point[1], color='blue', alpha=0.5, marker='o')
            plt.text(point[0], point[1], str(count), fontsize=8, ha='center', va='bottom')
        
        min_gen = min(map(int, data.keys()))
        max_gen = max(map(int, data.keys()))
        plt.xticks(np.arange(min_gen, max_gen + 1, 1))
        plt.xlim(min_gen - 1, max_gen + 1)
        plt.grid(True)
        plt.xlabel('Generation Number')
        plt.ylabel('Fitness Value')
        plt.title('Experiment Results')
        
        if not os.path.exists("./stat_results_vis_average_distance_with_obsts"):
            os.makedirs("./stat_results_vis_average_distance_with_obsts")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"./stat_results_vis_average_distance_with_obsts/{timestamp}.png")
        plt.close()
        
class SwarmSimulator(Simulator):
    simulate_single_call_count = 0
    _obst1_x_index = 0
    _obst1_y_index = 0
    _obst2_x_index = 0
    _obst2_y_index = 0

    @staticmethod
    def simulate(list_individuals, variable_names) -> List[SimulationOutput]:
        results = []
        for ind in list_individuals:
            simout = SwarmSimulator.simulate_single(ind, variable_names)
            results.append(simout)
        return results

    @staticmethod
    def simulate_single(vars, variable_names) -> SimulationOutput:
        SwarmSimulator._obst1_x_index = int(vars[1])
        SwarmSimulator._obst1_y_index = int(vars[2])
        SwarmSimulator._obst2_x_index = int(vars[3])
        SwarmSimulator._obst2_y_index = int(vars[4])
        SwarmSimulator.execute_swarm_rl(vars[0])
        log_file_path = os.path.join("projects", "quad_swarm_rl", "gym_art", "quadrotor_multi", "so.log")
        with open(log_file_path, 'r') as file:
            result3 = json.load(file)
        result_json = json.dumps(result3) 
        return SimulationOutput.from_json(result_json)

    def execute_swarm_rl(quads_obst_size):
        os.environ['OBST1_INDEX'] = str(8 * (SwarmSimulator._obst1_x_index - 1) + SwarmSimulator._obst1_y_index - 1)
        os.environ['OBST2_INDEX'] = str(8 * (SwarmSimulator._obst2_x_index - 1) + SwarmSimulator._obst2_y_index - 1)
        os.environ['quads_obst_size'] = str(quads_obst_size)
        os.environ['PATH_TO_SAVE'] = '../../pathOfDrones_average_distance_with_obsts'
        os.environ['OBST_SIZE'] = str(quads_obst_size)
        os.environ['GENERATION_NUM'] = str(SWARMProblem.counter)
        def run_command():
            command = [
                "python", "-m", "swarm_rl.enjoy",
                "--algo=APPO",
                "--env=quadrotor_multi",
                "--replay_buffer_sample_prob=0",
                "--quads_use_numba=False",
                "--train_dir=./train_dir",
                "--experiment=final",
                "--quads_view_mode=topdown",
                "--quads_render=False",
                "--no_render",
                "--quads_episode_duration=15.0",
                "--quads_mode=o_static_same_goal",
                f"--quads_obst_size={quads_obst_size}", 
            ]
            process = subprocess.Popen(command, cwd="./projects/quad_swarm_rl", env=os.environ)
            try:
                process.wait(timeout=1000000)  
            except subprocess.TimeoutExpired:
                process.kill()  
        run_command()

class MyFitnessFunction(Fitness):
    @property
    def min_or_max(self):
        return ["min", "min"]

    @property
    def name(self):
        return ["average_distance_with_obsts_drone0", "average_distance_with_obsts_drone7"]

    def eval(self, simout: SimulationOutput) -> Tuple[float]:
        return (simout.statistics["average_distance_with_obsts"][0], simout.statistics["average_distance_with_obsts_2"][0])

class MyCriticalFunction(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        return False

problem = SWARMProblem(
    problem_name="SwarmSimulatorProblem",
    xl=[0.75, 1, 3, 1, 3],
    xu=[1.5, 8.99, 6.99, 8.99, 6.99],
    simulation_variables=[
        "quads_obst_size", "quads_obst1_x_position", "quads_obst1_y_position", "quads_obst2_x_position", "quads_obst2_y_position"
    ],
    fitness_function=MyFitnessFunction(),
    critical_function=MyCriticalFunction(),
    simulate_function=SwarmSimulator.simulate
)

def main():
    log_utils.setup_logging("./log.txt")
    filename = 'stat_results_average_distance_with_obsts.txt'
    with open(filename, 'w') as file:
        file.write("")
    os.system("rm -rf ./pathOfDrones_average_distance_with_obsts")

    config = DefaultSearchConfiguration()
    config.n_generations = 2
    config.population_size = 2

    optimizer = NsgaIIOptimizer(problem=problem, config=config)

    try:
        optimizer.run()
        SWARMProblem.plot_generation_data(SWARMProblem.f_out)
        print(SWARMProblem.f_out)
    except RuntimeError:
        res = None
    
if __name__ == "__main__":
    main()