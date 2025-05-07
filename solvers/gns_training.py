from typing import Callable
from torch.nn import Module
from torch.optim import Adam
from model.replay_memory import Memories, Memory

def train(scheduling_agent: Module, embedding_stack: Module, shared_critic: Module, optimizer: Adam, memory: Memories, path: str, solve_function: Callable, device: str, run_number:int):
    pass