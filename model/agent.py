import pickle

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

class Agent:
    def __init__(self):
        self.policy: Module = None
        self.target: Module = None
        self.optimizer: Adam = None
        self.name: str = ""

    def save(self, path: str, version: int, itrs: int):
        torch.save(self.policy.state_dict(), f"{path}{self.name}_weights_{version}_{itrs}.pth")
        torch.save(self.optimizer.state_dict(), f"{path}{self.name}_optimizer_{version}_{itrs}.pth")

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
        self.name = ACTIONS_NAMES[OUTSOURCING]
        self.policy: L1_OutousrcingActor = L1_OutousrcingActor(shared_GNN, RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, AGENT_HIDDEN_DIM)
        self.target: L1_OutousrcingActor = L1_OutousrcingActor(shared_GNN, RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, AGENT_HIDDEN_DIM)
        self.target.load_state_dict(self.policy.state_dict())
        self._compile_and_move(device)

class SchedulingAgent(Agent):
    def __init__(self, shared_GNN: L1_EmbbedingGNN, device: str):
        self.name = ACTIONS_NAMES[SCHEDULING]
        self.policy: L1_SchedulingActor= L1_SchedulingActor(shared_GNN, RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, AGENT_HIDDEN_DIM)
        self.target: L1_SchedulingActor= L1_SchedulingActor(shared_GNN, RM_EMBEDDING_SIZE, OI_EMBEDDING_SIZE, AGENT_HIDDEN_DIM)
        self.target.load_state_dict(self.policy.state_dict())
        self._compile_and_move(device)

class MaterialAgent(Agent):
    def __init__(self, shared_GNN: L1_EmbbedingGNN, device: str):
        self.name = ACTIONS_NAMES[MATERIAL_USE]
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