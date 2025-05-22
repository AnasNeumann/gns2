from conf import *

import torch
from torch import Tensor
from model.gnn import State
import random 

# ########################################################
# =*= A REPLAY MEMORY DESTINGUISHING BETWEEN INSTANCES =*=
# ########################################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

class InstanceConfiguration:
    def __init__(self, alpha: Tensor, lb_Cmax: int, lb_cost: int, ub_Cmax: int, ub_cost: int):
        self.alpha: Tensor = alpha
        self.lb_Cmax: int = lb_Cmax
        self.lb_cost: int = lb_cost
        self.ub_Cmax: int = ub_Cmax
        self.ub_cost: int = ub_cost

class HistoricalState:
    def __init__(self, tree, state: State, possible_actions: list, end: int, cost: int, parent_action=None):
        self.state: State = state
        self.possible_actions: list = possible_actions
        self.actions_tested: list[Action] = []
        self.parent_action: Action = parent_action
        self.end: int = end
        self.cost: int = cost
        self.tree: Tree = tree
        if self.parent_action is not None:
            self.parent_action.next_state = self

class Action:
    def __init__(self, id: int, action_type: int, target: int, value: int, workload_removed: int=0, parent_state: HistoricalState=None):
        self.id: int                       = id
        self.action_type: int              = action_type
        self.target: int                   = target
        self.value: int                    = value
        self.exist_in_memory: bool         = False
        self.workload_removed: int         = workload_removed
        self.parent_state: HistoricalState = parent_state
        self.next_state: HistoricalState   = None
        self.reward: Tensor                = None
        if self.parent_state is not None:
            self.parent_state.actions_tested.append(self)

    def same(self, a) -> bool:
        a: Action
        return self.action_type == a.action_type and self.target == a.target and self.value == a.value
    
    # reward standardizator = STD_RATE * (α * ub_Cmax + (1-α) * ub_cost)
    def std_reward(self, conf: InstanceConfiguration) -> float:
        return ((conf.alpha * conf.ub_Cmax + (1-conf.alpha) * conf.ub_cost) * STD_RATE)
    
    # final reward = α * (Cmax - lb_Cmax) + (1-α) * (cost - lb_cost) / standardizator
    def final_reward(self, final_makespan: int, final_cost: int, std_reward: float, conf: InstanceConfiguration) -> float:
        makespan_final_expension: int = final_makespan - conf.lb_Cmax
        cost_final_expension: int     = final_cost - conf.lb_cost
        return (conf.alpha * makespan_final_expension + (1-conf.alpha) * cost_final_expension) / std_reward

    # step by step reward = α * (end_new - end_old - δ*workload_change) + (1-α) * (cost_new - cost_old) / standardizator
    def step_by_step_reward(self, std_reward: float, conf: InstanceConfiguration, w_workload: float=W_WORKLOAD) -> float:
        end_before: int      = self.parent_state.end if self.parent_state is not None else 0
        cost_before: int     = self.parent_state.cost if self.parent_state is not None else 0
        temporal_change: int = self.next_state.end - end_before - (w_workload*self.workload_removed)
        if self.action_type == OUTSOURCING and self.value == YES:
            cost_change: int = self.next_state.cost - cost_before
            return (conf.alpha * temporal_change + (1-conf.alpha) * cost_change) / std_reward
        return temporal_change / std_reward

    # Compute the final reward
    def compute_reward(self, conf: InstanceConfiguration, final_makespan: int, final_cost: int=-1, device: str="") -> Tensor:
        r_std: float          = self.std_reward(conf=conf)
        r_final: float        = self.final_reward(final_makespan=final_makespan, final_cost=final_cost, std_reward=r_std, conf=conf)
        r_step_by_step: float = self.step_by_step_reward(std_reward=r_std, conf=conf)
        reward: float         = W_FINAL * r_final + (1-W_FINAL) * r_step_by_step
        with torch.no_grad():
            self.reward = torch.tensor([-1.0 * reward], dtype=torch.float32, device=device)
        return self.reward

class Tree:
    """
        The memory of one specific instance (both pre-training and fine-tuning)
    """
    def __init__(self, global_memory, instance_id: str):
        self.instance_id: str = instance_id
        self.conf: InstanceConfiguration = None
        self.init_state: HistoricalState = None
        self.size: int = 0
        self.global_memory: Memory = global_memory

    def init_tree(self, alpha: Tensor, lb_Cmax: int, lb_cost: int, ub_Cmax: int, ub_cost: int):
        if self.conf == None:
            self.conf = InstanceConfiguration(alpha=alpha, lb_Cmax=lb_Cmax, lb_cost=lb_cost, ub_Cmax=ub_Cmax, ub_cost=ub_cost)

    # Compute all rewards of a new found branch of actions and states
    def compute_all_rewards(self, action: Action, final_makespan: int, final_cost: int=-1, device: str="") -> None:
        action.compute_reward(conf=self.conf, final_cost=final_cost, final_makespan=final_makespan, device=device)
        for _next in action.next_state.actions_tested:
            self.compute_all_rewards(action=_next, final_cost=final_cost, final_makespan=final_makespan, device=device)

    # Add decision in the memory or update reward if already exist
    def add_or_update_action(self, action: Action, final_makespan: int, final_cost: int=-1, need_rewards: bool=True, device: str="") -> Action:
        if need_rewards:
            self.compute_all_rewards(action=action, final_cost=final_cost, final_makespan=final_makespan, device=device)
        if action.parent_state is None:
            _found: bool = False
            for _other_first_action in self.init_state.actions_tested:
                if _other_first_action.same(action):
                    _found = True
                    _other_first_action.reward = torch.max(_other_first_action.reward, action.reward)
                    self.global_memory.add_action(_other_first_action)
                    for _next in action.next_state.actions_tested:
                        _next.parent_state = _other_first_action.next_state
                        self.add_or_update_action(action=_next, final_cost=final_cost, final_makespan=final_makespan, need_rewards=False, device=device)
                    return _other_first_action
            if not _found:
                action.parent_state = self.init_state
                self.init_state.actions_tested.append(action)
                self.size += 1
                _a: Action = action
                self.global_memory.add_action(_a)
                while _a.next_state.actions_tested:
                    _a = _a.next_state.actions_tested[0]
                    self.global_memory.add_action(_a)
                return action
        else:
            _found: bool = False
            for _existing_action in action.parent_state.actions_tested:
                if _existing_action.same(action):
                    _found = True
                    _existing_action.reward = torch.max(_existing_action.reward, action.reward)
                    self.global_memory.add_action(_existing_action)
                    for _next in action.next_state.actions_tested:
                        _next.parent_state = _existing_action.next_state
                        self.add_or_update_action(action=_next, final_cost=final_cost, final_makespan=final_makespan, need_rewards=False, device=device)
                    return _existing_action
            if not _found:
                action.parent_state.actions_tested.append(action)
                self.size += 1
                _a: Action = action
                self.global_memory.add_action(_a)
                while _a.next_state.actions_tested:
                    _a = _a.next_state.actions_tested[0]
                    self.global_memory.add_action(_a)
                return action
            
class _SegmentTree:
    """Sum-tree for priorities (power of two capacity for simplicity)."""
    def __init__(self, capacity: int):
        assert capacity & (capacity - 1) == 0, "cap must be power of 2"
        self.cap  = capacity
        self.tree = [0.0] * (2 * capacity)

    def _propagate(self, idx: int, change: float):
        while idx:
            idx //= 2
            self.tree[idx] += change

    def update(self, idx: int, new_p: float):
        """idx is data index (0-based)."""
        tree_idx = idx + self.cap
        change   = new_p - self.tree[tree_idx]
        self.tree[tree_idx] = new_p
        self._propagate(tree_idx, change)

    def total(self) -> float:
        return self.tree[1]

    def find_prefix(self, mass: float) -> int:
        idx = 1
        while idx < self.cap:
            if self.tree[2 * idx] >= mass:
                idx = 2 * idx
            else:
                mass -= self.tree[2 * idx]
                idx = 2 * idx + 1
        return idx - self.cap  # data index

class PrioritisedBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.alpha      = alpha
        self.capacity   = capacity
        self.pos        = 0
        self.size       = 0
        self.priorities = _SegmentTree(capacity)
        self.max_p      = 1.0
        self.data: list[Action] = [None]*capacity
        
    def append(self, act: Action):
        self.data[self.pos] = act
        self.priorities.update(self.pos, self.max_p**self.alpha)
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, beta: float):
        seg_total = self.priorities.total()
        step = seg_total / batch
        indices, probs = [], []
        for i in range(batch):
            mass = random.uniform(i*step, (i+1)*step)
            idx = self.priorities.find_prefix(mass)
            indices.append(idx)
            probs.append(self.priorities.tree[idx + self.priorities.cap])
        probs = torch.tensor(probs, dtype=torch.float32)
        sampling_p = probs / seg_total
        weights = (self.size * sampling_p).pow(-beta)
        weights /= weights.max()
        actions = [self.data[i] for i in indices]
        return actions, indices, weights

    def update_priorities(self, indices, new_p):
        for i, p in zip(indices, new_p):
            p = float(p)
            self.priorities.update(i, p ** self.alpha)
            self.max_p = max(self.max_p, p)

class Memory:
    def __init__(self, cap: int = MEMORY_CAPACITY, alpha: float = 0.6, beta0: float = 0.4, beta_steps: int = 2_000_000):
        self.instances: list[Tree] = []
        self.beta0, self.beta_steps = beta0, beta_steps
        self.n_updates  = 0
        self.buff_out   = PrioritisedBuffer(cap, alpha)
        self.buff_sched = PrioritisedBuffer(cap, alpha)
        self.buff_mat   = PrioritisedBuffer(cap, alpha)
        self.flat_memories = [self.buff_out, self.buff_sched, self.buff_mat]

    #  Add a new instance if ID is not present yet
    def add_instance_if_new(self, id: int) -> Tree:
        for memory in self.instances:
            if memory.instance_id == id:
                return memory
        new_memory: Tree = Tree(global_memory=self, instance_id=id)
        self.instances.append(new_memory)
        return new_memory

    def add_action(self, action: Action):
        if not action.exist_in_memory:
            self.flat_memories[action.action_type].append(action)
            action.exist_in_memory = True
    
    # Linear annealing β⟶1
    def _beta(self):
        return min(1.0, self.beta0 + (1.0 - self.beta0) * (self.n_updates / self.beta_steps))

    def sample(self, action_type: int, batch: int):
        beta = self._beta()
        actions, idxs, w = self.flat_memories[action_type].sample(batch, beta)
        return actions, idxs, w

    def update_priorities(self, action_type: int, idxs, td_errors):
        prios = td_errors.view(-1).tolist()
        self.flat_memories[action_type].update_priorities(idxs, prios)
        self.n_updates += 1