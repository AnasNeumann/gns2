import pickle
import os

from model.agent import Agents
from model.instance import Instance

from conf import *
from tools.common import directory

# ##################################################
# =*= GNN TRAINER USING E-GREEDY DEEP Q-LEARNING =*=
# ##################################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

# Main function of the e-greedy DQN trainer 
def train(agents: Agents, path: str, device: str, version:int, itrs: int, debug:bool):
    pass