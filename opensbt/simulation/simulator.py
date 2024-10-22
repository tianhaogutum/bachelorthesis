
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
from pymoo.core.individual import Individual
from abc import ABC, abstractmethod, abstractstaticmethod
from opensbt.utils.encoder_utils import NumpyEncoder

import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

'''
    Represent data output by a simulator

    Example json representation of a SimulationOutput instance:

    {
        "simTime" : 3,
        "times": [1.0,2.0,3.0],
        "location": { "ego" : [(1,1),(2,2),(3,3)],
                    "adversary" : [(4,1),(4,2),(4,3)},

        "velocity": { "ego" : [0.5,0.5,0.5],
                        "adversary" : [0.9,0.9,0.9],
                        },
        "collisions": [],
        "actors" : {1: "ego",
                        2: "adversary"
                    },
        "otherParams" : {
            "isCollision": False
        }
    }
        
'''
from dataclasses import dataclass
from typing import Dict, List
import json

@dataclass
class SimulationOutput:
    distance_log: Dict[str, List[float]]
    collisions_log: Dict[str, List[int]]
    statistics: Dict[str, List[float]]
    otherParams: Dict

    def __init__(self, distance_log=None, collisions_log=None, statistics=None, otherParams=None, **kwargs):
        self.distance_log = distance_log if distance_log is not None else {}
        self.collisions_log = collisions_log if collisions_log is not None else {}
        self.statistics = statistics if statistics is not None else {}
        self.otherParams = otherParams if otherParams is not None else {}
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_json(self):
        return json.dumps(self.__dict__, allow_nan=True, indent=4)

    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)
        return cls(**json_dict)



class Simulator(ABC):
        
    @abstractstaticmethod
    def simulate(list_individuals: List[Individual], 
                variable_names: List[str], 
                scenario_path: str, 
                sim_time: float = 10, 
                time_step: float = 0.01, 
                do_visualize: bool = True) -> List[SimulationOutput]:
        pass