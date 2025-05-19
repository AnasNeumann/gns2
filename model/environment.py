from model.graph import GraphInstance
from model.instance import Instance
from model.replay_memory import Action

from torch import Tensor
from copy import deepcopy

# ########################################################
# =*= A REPLAY MEMORY DESTINGUISHING BETWEEN INSTANCES =*=
# ########################################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

class Environment:
    def __init__(self, instance: Instance, graph: GraphInstance, lb_Cmax: int, ub_Cmax: int, lb_cost: int, ub_cost: int, alpha: Tensor=None):
        self.i: Instance = instance
        self.graph: GraphInstance = graph
        self.possible_actions: list[(int, int)] = []
        self.action_type: int = -1
        self.action_id: int = -1
        self.execution_time: int = -1
        self.past_envs: list[Environment] = []
        self.lb_Cmax: int = lb_Cmax
        self.ub_Cmax: int = ub_Cmax
        self.lb_cost: int = lb_cost
        self.ub_cost: int = ub_cost
        self.alpha: Tensor = alpha
        self.Q: Queue = None
        self.previous_actions: list[Action] = []
        self.current_cmax: int = 0
        self.current_cost: int = 0
        
    def clone(self):
        env: Environment = Environment(self.i, self.graph.clone(), self.lb_Cmax, self.ub_Cmax, self.lb_cost, self.ub_cost, self.alpha) 
        env.current_cmax = self.current_cmax
        env.current_cost = self.current_cost
        env.past_envs = self.past_envs.copy()
        env.past_envs.append(self)
        env.Q = self.Q.clone()
        _branch: Action = self.previous_actions[0].clone()
        env.previous_actions.append(_branch)
        while _branch.next_state.actions_tested:
            _branch = _branch.next_state.actions_tested[0]
            env.previous_actions.append(_branch)
        return env
    
    def action_found(self, possible_actions: list[(int, int)], action_type: int, action_id: int, execution_time: int):
        self.possible_actions = possible_actions
        self.action_type = action_type
        self.action_id = action_id
        self.execution_time = execution_time

    def get_last_action(self):
        return self.actions[-1] if len(self.actions) > 0 else None
    
    def get_base_action(self):
        return self.actions[0] if len(self.actions) > 0 else None
    
    def is_first_env(self):
        return len(self.actions) == 0
        
    # Init the task and time queue
    def init_queue(self):
        for item_id in self.graph.project_heads:
            p, head = self.graph.items_g2i[item_id]
            for o in self.i.first_operations(p, head):
                self.Q.add_operation(self.graph.operations_i2g[p][o])

class Queue:
    def __init__(self):
        self.operation_queue: list[int] = []
        self.item_queue: list[int] = []

    def clone(self):
        q: Queue = Queue()
        q.operation_queue = deepcopy(self.operation_queue)
        q.item_queue(self.item_queue)
        return q

    def done(self) -> bool:
        return len(self.operation_queue) == 0 and len(self.item_queue) == 0 # and len(self.item_queue) == 0

    def add_operation(self, operation: int):
        self.operation_queue.append(operation)

    def add_item(self, item: int):
        self.item_queue.append(item)
    
    def remove_operation(self, operation: int):
        self.operation_queue.remove(operation)  
    
    def remove_item(self, item: int):
        self.item_queue.remove(item) 