import pickle
import random
import matplotlib.pyplot as plt

from torch.nn import Module
from torch.optim import Adam

from model.fast_gnn import *
from model.replay_memory import Memory, Action, HistoricalState
from conf import *

from tools.common import directory

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
    def __init__(self, multi_agents_system, name: str, color: str, action_type: int, device: str, interactive: bool):
        self.multi_agents_system: Agents = multi_agents_system
        self.policy: Module = None
        self.target: Module = None
        self.optimizer: Adam = None
        self.name: str = name
        self.device: str = device
        self.action_type: int = action_type
        self.loss: Loss = Loss(xlabel="Episode", ylabel="Loss", title="Huber Loss ("+self.name+" policy network)", color=color, show=interactive)

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
        # self.policy    = torch.compile(self.policy, backend="aot_eager", dynamic=True, fullgraph=False)
        # self.target    = torch.compile(self.target, backend="aot_eager", dynamic=True, fullgraph=False)
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
        loss_accum: float = 0
        for action in actions:
            history: HistoricalState = action.parent_state
            _alpha: Tensor           = history.tree.alpha
            Q_logits: Tensor         = self.policy(history.state, history.possible_actions, _alpha) # Qπ(s,a)
            Q_sa: Tensor             = Q_logits[action.id]
            done: bool               = len(action.next_state.possible_actions) == 0
            if done:
                target_val = action.reward
            else:
                next_state: HistoricalState = action.next_state
                next_target_agent: Module   = self.multi_agents_system.agents[next_state.actions_tested[0].action_type].target
                with torch.no_grad(): # max_a′ Q_target_{who_acts_next}(s′,a′)
                    Q_next_logits: Tensor   = next_target_agent(next_state.state, next_state.possible_actions, _alpha)
                max_Q_next: Tensor          = Q_next_logits.max()
                target_val: Tensor          = action.reward + GAMMA * max_Q_next
            huber_loss = F.smooth_l1_loss(Q_sa, target_val, reduction="mean", beta=DELTA)
            huber_loss.backward()
            loss_accum += huber_loss.item()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        return loss_accum / len(actions)

class OustourcingAgent(Agent):
    def __init__(self, multi_agents_system, device: str, interactive: bool):
        super().__init__(multi_agents_system, ACTIONS_NAMES[OUTSOURCING], ACTIONS_COLOR[OUTSOURCING], OUTSOURCING, device, interactive)
        self.policy: OutousrcingActor = OutousrcingActor()
        self.target: OutousrcingActor = OutousrcingActor()
        self.target.load_state_dict(self.policy.state_dict())
        self._compile_and_move()

class SchedulingAgent(Agent):
    def __init__(self, multi_agents_system, device: str, interactive: bool):
        super().__init__(multi_agents_system, ACTIONS_NAMES[SCHEDULING], ACTIONS_COLOR[SCHEDULING], SCHEDULING, device, interactive)
        self.policy: SchedulingActor= SchedulingActor()
        self.target: SchedulingActor= SchedulingActor()
        self.target.load_state_dict(self.policy.state_dict())
        self._compile_and_move()

class MaterialAgent(Agent):
    def __init__(self, multi_agents_system, device: str, interactive: bool):
        super().__init__(multi_agents_system, ACTIONS_NAMES[MATERIAL_USE], ACTIONS_COLOR[MATERIAL_USE], MATERIAL_USE, device, interactive)
        self.policy: MaterialActor = MaterialActor()
        self.target: MaterialActor = MaterialActor()
        self.target.load_state_dict(self.policy.state_dict())
        self._compile_and_move()
    
class Agents:
    def __init__(self, device: str, base_path: str, version: int=-1, interactive: bool=False):
        self.device: str = device
        self.base_path: str = base_path+directory.models+'/'
        self.version: int = version
        self.memory: Memory = Memory()
        self.agents: list[Agent] = [OustourcingAgent(self, self.device, interactive), 
                                    SchedulingAgent(self, self.device, interactive), 
                                    MaterialAgent(self, self.device, interactive)]

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
        for i, agent in enumerate(self.agents):
            if self.memory.flat_memories[i]:
                print(f"\t -> optimizing {agent.name} policy network...")
                l: float = agent.optimize_policy(self.memory)
                print(f"\t -> optimizing {agent.name} target network...")
                agent.optimize_target()
                agent.loss.update(loss_value=l)
                losses.append(l)
            else:
                losses.append(0.0)
                print(f"\t -> No need to optimize {agent.name} policy network (replay memory still empty)...")
        return losses