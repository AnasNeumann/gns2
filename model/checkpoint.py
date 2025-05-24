from conf import *
from tools.common import objective_value

# #######################################
# =*= A CHECKPOINT FOR THE GNS SOLVER =*=
# #######################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

class Checkpoint:
    def __init__(self, start_at: int=0):
        self.cmax: int                 = start_at
        self.cost: int                 = start_at
        self.banned_step: int          = start_at
        self.decision_made: list[int]  = []
        self.nb_choices: list[int]     = []
        self.decisions_type: list[int] = []

    def add(self, actions: list, type: int, id: int):
        self.nb_choices.append(len(actions))
        self.decision_made.append(id)
        self.decisions_type.append(type)

    def next_step(self) -> int:
        self.banned_step +=1
        while self.banned_step < len(self.decision_made) and (self.nb_choices[self.banned_step] < 2 or self.decisions_type[self.banned_step] == MATERIAL_USE):
            self.banned_step +=1
        if self.banned_step >= len(self.decision_made) or self.nb_choices[self.banned_step] < 2:
            return -1
        return self.banned_step
    
    def is_better_than(self, o: 'Checkpoint', w_makespan: float) -> bool:
        return objective_value(self.cmax, self.cost, w_makespan) <= objective_value(o.cmax, o.cost, w_makespan) or o.cmax < 0
