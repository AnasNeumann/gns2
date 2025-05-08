import pickle
import matplotlib.pyplot as plt

from torch.nn import Module
from torch.optim import Adam

from model.gnn import *
from model.replay_memory import Memory
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
    def __init__(self, name: str, color: str):
        self.policy: Module = None
        self.target: Module = None
        self.optimizer: Adam = None
        self.name: str = name
        self.loss: Loss = Loss(xlabel="Episode", ylabel="Loss", title="Huber Loss ("+self.name+" policy network)", color=color, show=INTERACTIVE)

    def save(self, path: str, version: int, itrs: int):
        torch.save(self.policy.state_dict(), f"{path}{self.name}_weights_{version}_{itrs}.pth")
        torch.save(self.optimizer.state_dict(), f"{path}{self.name}_optimizer_{version}_{itrs}.pth")
        self.loss.save(f"{path}{self.name}_loss")

    def load(self, path: str, version: int, itrs: int, device: str):
        self.policy.load_state_dict(torch.load(f"{path}{self.name}_weights_{version}_{itrs}.pth", map_location=torch.device(device), weights_only=True))
        self.optimizer.load_state_dict(torch.load(f"{path}{self.name}_optimizer_{version}_{itrs}.pth", map_location=torch.device(device), weights_only=True))

    def _compile_and_move(self, device: str):
        self.policy.to(device=device)
        self.target.to(device=device)
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

    def optimize_policy(self):
        pass

class OustourcingAgent(Agent):
    def __init__(self, shared_GNN: L1_EmbbedingGNN, device: str):
        super().__init__(ACTIONS_NAMES[OUTSOURCING], ACTIONS_COLOR[OUTSOURCING])
        self.policy: L1_OutousrcingActor = L1_OutousrcingActor(shared_GNN, RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, AGENT_HIDDEN_DIM)
        self.target: L1_OutousrcingActor = L1_OutousrcingActor(shared_GNN, RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, AGENT_HIDDEN_DIM)
        self.target.load_state_dict(self.policy.state_dict())
        self._compile_and_move(device)

class SchedulingAgent(Agent):
    def __init__(self, shared_GNN: L1_EmbbedingGNN, device: str):
        super().__init__(ACTIONS_NAMES[SCHEDULING], ACTIONS_COLOR[SCHEDULING])
        self.policy: L1_SchedulingActor= L1_SchedulingActor(shared_GNN, RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, AGENT_HIDDEN_DIM)
        self.target: L1_SchedulingActor= L1_SchedulingActor(shared_GNN, RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, AGENT_HIDDEN_DIM)
        self.target.load_state_dict(self.policy.state_dict())
        self._compile_and_move(device)

class MaterialAgent(Agent):
    def __init__(self, shared_GNN: L1_EmbbedingGNN, device: str):
        super().__init__(ACTIONS_NAMES[MATERIAL_USE], ACTIONS_COLOR[MATERIAL_USE])
        self.policy: L1_MaterialActor = L1_MaterialActor(shared_GNN, RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, AGENT_HIDDEN_DIM)
        self.target: L1_MaterialActor = L1_MaterialActor(shared_GNN, RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, AGENT_HIDDEN_DIM)
        self.target.load_state_dict(self.policy.state_dict())
        self._compile_and_move(device)
    
class Agents:
    def __init__(self, device: str, base_path: str, version: int=-1):
        self.device: str = device
        self.base_path: str = base_path
        self.version: int = version
        self.memory: Memory = Memory()
        self.shared_GNN: L1_EmbbedingGNN = L1_EmbbedingGNN(RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, EMBEDDING_HIDDEN_DIM, STACK_SIZE)
        self.agents: list[Agent] = [OustourcingAgent(self.shared_GNN, device), SchedulingAgent(self.shared_GNN, device), MaterialAgent(self.shared_GNN, device)]

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