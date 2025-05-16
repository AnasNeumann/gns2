import time as systime
import random

from solvers.gns_solver import solve

from model.agent import Agents
from model.instance import Instance
from model.dataset import Dataset
from model.replay_memory import Tree

from conf import *
from tools.common import directory

# ##################################################
# =*= GNN TRAINER USING E-GREEDY DEEP Q-LEARNING =*=
# ##################################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

def get_instance_and_tree(dataset: Dataset, agents: Agents, episode: int=0, instance: Instance=None, tree: Tree=None):
    if instance is None or episode % SWITCH_INSTANCE == 0:
        print("...time to switch the instance!")
        instance: Instance = dataset.random_one()
        tree: Tree = agents.memory.add_instance_if_new(id=instance.id)
        return instance, tree
    return instance, tree

# Main function of the e-greedy DQN trainer 
def train(agents: Agents, path: str, device: str, version:int, itrs: int, debug:bool):
    _start_time = systime.time()
    print("Loading dataset....")
    dataset: Dataset = Dataset(path)
    dataset.load_training_instances(version)
    instance, tree = get_instance_and_tree(dataset, agents)
    for episode in range(1, EPISODES+1):
        instance, tree = get_instance_and_tree(dataset, agents, episode, instance, tree)
        _greedy = random.random() <= 0.85
        solve(instance=instance, agents=agents, train=True, device=device, greedy=_greedy, REPLAY_MEMORY=tree, episode=episode, debug=debug)
        if episode % TRAINING_ITRS == 0: 
            hl: list[float] = agents.optimize()
            print(f"Training episode: {episode} [time={(systime.time()-_start_time):.2f}] -- instance: ({instance.size}, {instance.id}) -- Outsourcing Loss: {hl[OUTSOURCING]:.2f} -- Scheduling Loss: {hl[SCHEDULING]:.2f} -- Material Use Loss: {hl[MATERIAL_USE]:.2f}")
        else:
            print(f"Training episode: {episode} [time={(systime.time()-_start_time):.2f}] -- instance: ({instance.size}, {instance.id}) -- No optimization yet...")
        if episode % SAVING_ITRS == 0 or episode == EPISODES: 
            print("...time to save current policy models, optimizers, and the replay memory!")
            agents.save(itrs + episode)
    print("--- =*= END_OF_FILE =*= ---")