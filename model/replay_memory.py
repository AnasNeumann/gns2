from conf import *
from torch import Tensor
from model.gnn import State

# ########################################################
# =*= A REPLAY MEMORY DESTINGUISHING BETWEEN INSTANCES =*=
# ########################################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

class HistoricalState:
    def __init__(self, state: State, possible_actions: list, end: int, cost: int, parent_action=None):
        self.state = state
        self.possible_actions = possible_actions
        self.actions_tested: list[Action] = []
        self.parent_action: Action = parent_action
        self.end = end
        self.cost = cost
        if self.parent_action.next_state is not None:
            self.parent_action.next_state = self

class Action:
    def __init__(self, id: int, action_type: int, target: int, value: int, parent_state: HistoricalState=None):
        self.id: int = id
        self.action_type: int = action_type
        self.target: int = target
        self.value: int = value
        self.parent_state: HistoricalState = parent_state
        self.next_state: HistoricalState = None
        self.reward: float = -1
        if self.parent_state is not None:
            self.parent_state.actions_tested.append(self)

    def same(self, a) -> bool:
        a: Action
        return self.parent_state == a.parent_state and self.action_type == a.action_type and self.target == a.target and self.value == a.value
    
    # Compute the final reward
    def compute_reward(self, a: float, init_cmax: int, init_cost: int, final_makespan: int, final_cost: int=-1) -> float:
        _d: float = STD_RATE*(a*init_cmax + (1-a)*init_cost)
        makespan_part: float =  (1.0-W_FINAL) * (self.end_new - self.end_old) + W_FINAL * (final_makespan - init_cmax)
        if self.action_type == OUTSOURCING:
            cost_part: float = (1.0-W_FINAL) * (self.cost_new - self.cost_old) + W_FINAL * (final_cost - init_cost)
            self.reward = -1.0 * (a*makespan_part + (1-a)*cost_part)/_d
        else:
            self.reward = -1.0 * (a*makespan_part)/_d
        return self.reward

class Tree:
    """
        The memory of one specific instance (both pre-training and fine-tuning)
    """
    def __init__(self, global_memory, instance_id: str):
        self.instance_id: str = instance_id
        self.alpha: Tensor = None
        self.init_makesan: int = -1
        self.init_cost: int = -1
        self.init_state: HistoricalState = None
        self.size: int = 0
        self.global_memory: Memory = global_memory

    def init_tree(self, init_state: HistoricalState, alpha: Tensor, init_makesan: int, init_cost: int):
        if self.init_state == None:
            self.init_state = init_state
            self.init_makesan = init_makesan
            self.init_cost = init_cost
            self.alpha = alpha

    # Compute all rewards of a new found branch of actions and states
    def compute_all_rewards(self, action: Action, final_makespan: int, final_cost: int=-1) -> None:
        action.compute_reward(a=self.alpha.item(), final_cost=final_cost, final_makespan=final_makespan, init_cmax=self.init_makesan, init_cost=self.init_cost)
        for _next in action.next_state.actions_tested:
            self.compute_all_rewards(action=_next, final_cost=final_cost, final_makespan=final_makespan)

    # Add decision in the memory or update reward if already exist
    def add_or_update_action(self, action: Action, final_makespan: int, final_cost: int=-1, need_rewards: bool=True) -> Action:
        if need_rewards:
            self.compute_all_rewards(decision=action, final_cost=final_cost, final_makespan=final_makespan)
        if action.parent_state is None:
            _found: bool = False
            for _other_first_action in self.init_state.actions_tested:
                if _other_first_action.same(action):
                    _found = True
                    _other_first_action.reward = max(_other_first_action.reward, action.reward)
                    for _next in action.next_state.actions_tested:
                        _next.parent_state = _other_first_action.next_state
                        self.add_or_update_action(action=_next, final_cost=final_cost, final_makespan=final_makespan, need_rewards=False)
                    return _other_first_action
            if not _found:
                self.init_state.actions_tested(action)
                self.size += 1
                self.global_memory.add_into_flat_memory(action)
                _t: Action = action
                while _t.next_state.actions_tested:
                    self.global_memory.add_into_flat_memory(_t)
                    _t = _t.next_state.actions_tested[0]
                return action
        else:
            _found: bool = False
            for _existing_action in action.parent_state.actions_tested:
                if _existing_action.same(action):
                    _found = True
                    _existing_action.reward = max(_next.reward, action.reward)
                    for _next in action.next_state.actions_tested:
                        _next.parent_state = _existing_action.next_state
                        self.add_or_update_action(action=_next, final_cost=final_cost, final_makespan=final_makespan, need_rewards=False)
                    return _existing_action
            if not _found:
                action.parent_state.actions_tested.append(action)
                self.size += 1
                self.global_memory.add_into_flat_memory(action)
                _t: Action = action
                while _t.next_state.actions_tested:
                    self.global_memory.add_into_flat_memory(_t)
                    _t = _t.next_state.actions_tested[0]
                return action

class Memory:
    """
        The memory for several instances (pre-training time only)
    """
    def __init__(self):
        self.instances: list[Tree] = []
        self.flat_memory: list[Action] = []

    def add_into_flat_memory(self, action: Action):
        self.flat_memory.append(action)
        if len(self.flat_memory) > MEMORY_CAPACITY:
            self.flat_memory.pop(0)
    
    #  Add a new instance if ID is not present yet
    def add_instance_if_new(self, id: int) -> Tree:
        for memory in self.instances:
            if memory.instance_id == id:
                return memory
        new_memory: Tree = Tree(instance_id=id)
        self.instances.append(new_memory)
        return new_memory
