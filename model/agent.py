import pickle
import random
import matplotlib.pyplot as plt

from torch.nn import Module
from torch.optim import Adam

from model.gnn import *
from model.replay_memory import Memory, Action, HistoricalState
from conf import *

# #############################################
# =*= AI AGENTS DATACLASSES AND DRL METHODS =*=
# #############################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

class Loss():
    def __init__(self, xlabel: str, ylabel: str, title: str, color: str, show: bool = True, width=7.04, height=4.80):
        self.show = show
        if self.show:
            plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(width, height))
        self.x_data = []
        self.y_data = []
        self.episode = 0
        self.line, = self.ax.plot(self.x_data, self.y_data, label=title, color=color)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.legend()
        if self.show:
            plt.ioff()
    
    def update(self, loss_value: float):
        self.episode = self.episode + 1
        self.x_data.append(self.episode)
        self.y_data.append(loss_value)
        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        if self.show:
            plt.pause(0.0001)

    def save(self, filepath: str):
        self.fig.savefig(filepath + ".png")
        with open(filepath + '_x_data.pkl', 'wb') as f:
            pickle.dump(self.x_data, f)
        with open(filepath + '_y_data.pkl', 'wb') as f:
            pickle.dump(self.y_data, f)

class Agent:
    def __init__(self, multi_agents_system, name: str, action_type: int, color: str, device: str):
        self.multi_agents_system: Agents = multi_agents_system
        self.policy: Module = None
        self.target: Module = None
        self.optimizer: Adam = None
        self.name: str = name
        self.device: str = device
        self.action_type: int = action_type
        self.loss: Loss = Loss(xlabel="Episode", ylabel="Loss", title="Huber Loss ("+self.name+" policy network)", color=color, show=INTERACTIVE)

    def save(self, path: str, version: int, itrs: int):
        torch.save(self.policy.state_dict(), f"{path}{self.name}_weights_{version}_{itrs}.pth")
        torch.save(self.optimizer.state_dict(), f"{path}{self.name}_optimizer_{version}_{itrs}.pth")
        self.loss.save(f"{path}{self.name}_loss")

    def load(self, path: str, version: int, itrs: int, device: str):
        self.policy.load_state_dict(torch.load(f"{path}{self.name}_weights_{version}_{itrs}.pth", map_location=torch.device(device), weights_only=True))
        self.optimizer.load_state_dict(torch.load(f"{path}{self.name}_optimizer_{version}_{itrs}.pth", map_location=torch.device(device), weights_only=True))

    def _compile_and_move(self):
        self.policy.to(device=self.device)
        self.target.to(device=self.device)
        self.policy.train()
        self.target.train()
        torch.compile(self.policy)
        torch.compile(self.target)
        self.optimizer = Adam(list(self.policy.parameters()), lr=LEARNING_RATE)

    def optimize_target(self):
        _target_weights = self.target.state_dict()
        _policy_weights = self.policy.state_dict()
        for param in _policy_weights:
            _target_weights[param] = _policy_weights[param] * TAU + _target_weights[param] * (1 - TAU)
        self.target.load_state_dict(_target_weights)

    def optimize_policy(self, replay_memory: Memory) -> float:
        """
            Optimize the polict network using the Huber loss between selected action and expected best action (based on approx Q-value)
                y = reward r + discounted factor γ x MAX_Q_VALUES(state s+1) predicted with Q_target
                x = predicted quality of (s, a) using the policy network
                L(x, y) = 1/2 (x-y)^2 for small errors (|x-y| ≤ δ) else δ|x-y| - 1/2 x δ^2
        """
        memory: list[Action] = replay_memory.flat_memories[self.action_type]
        if len(memory) == 0:
            return 0.0
        actions: list[Action] = random.sample(memory, min(BATCH_SIZE, len(memory)))
        self.optimizer.zero_grad(set_to_none=True)
        loss_accum: Tensor = torch.zeros(1, device=self.device)
        for action in actions:
            history: HistoricalState = action.parent_state
            _alpha: Tensor           = history.tree.alpha
            _related_items: Tensor   = history.tree.related_items
            _parent_items: Tensor    = history.tree.parents
            Q_logits    = self.policy(history.state, history.possible_actions, _related_items, _parent_items, _alpha) # Qπ(s,a)
            Q_sa        = Q_logits[action.id]
            done: bool  = len(action.next_state.possible_actions) == 0
            with torch.no_grad(): # max_a′ Q_target_{who_acts_next}(s′,a′)
                if done:
                    target_val = action.reward
                else:
                    next_state: HistoricalState = action.next_state
                    next_target_agent: Module   = self.multi_agents_system.agents[next_state.actions_tested[0].action_type].target
                    with torch.no_grad():
                        Q_next_logits: Tensor   = next_target_agent(next_state.state, next_state.possible_actions, _related_items, _parent_items, _alpha)
                    max_Q_next                  = Q_next_logits.max()
                    target_val                  = action.reward + GAMMA * max_Q_next
            huber_loss = F.smooth_l1_loss(Q_sa, target_val, reduction='none')
            loss_accum += huber_loss
        mean_loss = loss_accum.mean()
        mean_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        return mean_loss.item()

class OustourcingAgent(Agent):
    def __init__(self, multi_agents_system, shared_GNN: L1_EmbbedingGNN, device: str):
        super().__init__(multi_agents_system, ACTIONS_NAMES[OUTSOURCING], ACTIONS_COLOR[OUTSOURCING], OUTSOURCING, device)
        self.policy: L1_OutousrcingActor = L1_OutousrcingActor(shared_GNN, RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, AGENT_HIDDEN_DIM)
        self.target: L1_OutousrcingActor = L1_OutousrcingActor(shared_GNN, RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, AGENT_HIDDEN_DIM)
        self.target.load_state_dict(self.policy.state_dict())
        self._compile_and_move()

class SchedulingAgent(Agent):
    def __init__(self, multi_agents_system, shared_GNN: L1_EmbbedingGNN, device: str):
        super().__init__(multi_agents_system, ACTIONS_NAMES[SCHEDULING], ACTIONS_COLOR[SCHEDULING], SCHEDULING, device)
        self.policy: L1_SchedulingActor= L1_SchedulingActor(shared_GNN, RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, AGENT_HIDDEN_DIM)
        self.target: L1_SchedulingActor= L1_SchedulingActor(shared_GNN, RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, AGENT_HIDDEN_DIM)
        self.target.load_state_dict(self.policy.state_dict())
        self._compile_and_move()

class MaterialAgent(Agent):
    def __init__(self, multi_agents_system, shared_GNN: L1_EmbbedingGNN, device: str):
        super().__init__(multi_agents_system, ACTIONS_NAMES[MATERIAL_USE], ACTIONS_COLOR[MATERIAL_USE], MATERIAL_USE, device)
        self.policy: L1_MaterialActor = L1_MaterialActor(shared_GNN, RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, AGENT_HIDDEN_DIM)
        self.target: L1_MaterialActor = L1_MaterialActor(shared_GNN, RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, AGENT_HIDDEN_DIM)
        self.target.load_state_dict(self.policy.state_dict())
        self._compile_and_move()
    
class Agents:
    def __init__(self, device: str, base_path: str, version: int=-1):
        self.device: str = device
        self.base_path: str = base_path
        self.version: int = version
        self.memory: Memory = Memory()
        self.shared_GNN: L1_EmbbedingGNN = L1_EmbbedingGNN(RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, EMBEDDING_HIDDEN_DIM, STACK_SIZE)
        self.agents: list[Agent] = [OustourcingAgent(self, self.shared_GNN, device), SchedulingAgent(self, self.shared_GNN, device), MaterialAgent(self, self.shared_GNN, device)]

    def load(self, itrs: int):
        for agent in self.agents:
            agent.load(self.base_path, self.version, itrs, self.device)
        with open(f"{self.base_path}memory_{self.version}_{itrs}.pth", 'rb') as file:
            self.memory: Memory = pickle.load(file)

    def save(self, itrs: int):
        for agent in self.agents:
            agent.save(self.base_path, self.version, itrs)
        with open(f"{self.base_path}memory_{self.version}_{itrs}.pth", 'wb') as f:
            pickle.dump(self.memory, f)

    def optimize(self) -> list[float]:
        losses: list[float] = []
        for agent in self.agents:
            l: float = agent.optimize_policy()
            agent.optimize_target()
            agent.loss.update(loss_value=l)
            losses.append(l)
        return losses