import argparse
import time as systime

import pandas as pd
import torch
torch.autograd.set_detect_anomaly(True)

from conf import *
from tools.common import to_bool, directory, objective_value

from solvers.gns_solver import solve
from solvers.gns_training import train as pre_train

from model.instance import Instance
from model.agent import Agents
from model.dataset import Dataset

# ###################################################################
# =*= MAIN FILE OF THE PROJECT: CHOOSE TO TRAIN OR TEST THE MODEL =*=
# ###################################################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

# Pre-train networks on all instances
def train(version: int, itrs: int, agents: Agents, device: str, path: str, debug: bool):
    print("Pre-training models with e-greedy DQN (on several instances)...")
    pre_train(agents=agents, path=path, device=device, version=version, itrs=itrs, debug=debug)
    
# Solve the target instance (size, id) only using inference
def solve_one_instance(instance: Instance, size: str, agents: Agents, device: str, path: str, debug: bool=False):
    start_time = systime.time()
    best_cmax = -1.0
    best_cost = -1.0
    best_obj = -1.0
    print(f"SOLVING INSTANCE {size}_{instance.id}...")
    current_cmax, current_cost = solve(instance, agents=agents, train=False, device=device, debug=debug)
    _obj = objective_value(current_cmax, current_cost, instance.w_makespan)/100
    if best_obj < 0 or _obj < best_obj:
        best_obj = _obj
        best_cmax = current_cmax
        best_cost = current_cost
    final_metrics = pd.DataFrame({
        'index': [instance.id],
        'value': [best_obj],
        'cmax': [best_cmax],
        'cost': [best_cost],
        'computing_time': [systime.time()-start_time],
        'device_used': [device]
    })
    print(final_metrics)
    final_metrics.to_csv(path+directory.instances+'/test/'+size+'/solution_gns_'+str(instance.id)+'.csv', index=False)
    return instance

# Solve all instances only in inference mode
def solve_all_instances(agents: Agents, version: int, device: str, path: str, debug:bool):
    dataset: Dataset = Dataset(base_path=path)
    dataset.load_test_instances()
    for i in dataset.test_instances:
        solve_one_instance(instance=i, size=str(i.size), agents=agents, device=device, path=path, debug=debug)

def build_agents(device: str, version: int, itrs: int, path: str, interactive: bool):
    agents: Agents = Agents(device=device, base_path=path, version=version, interactive=interactive)
    if version > 1 or itrs > 0:
        agents.load(itrs)
    return agents

# MAIN CODE STARTING POINT ##############################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII/L1 GNS solver")
    parser.add_argument("--size", help="Size of the solved instance", required=False)
    parser.add_argument("--id", help="Id of the solved instance", required=False)
    parser.add_argument("--train", help="Do you want to load a pre-trained model", required=True)
    parser.add_argument("--target", help="Do you want to load a pre-trained model", required=False)
    parser.add_argument("--mode", help="Execution mode (either prod or test)", required=True)
    parser.add_argument("--path", help="Saving path on the server", required=True)
    parser.add_argument("--use_pretrain", help="Use a pre-train model while fine-tuning", required=False)
    parser.add_argument("--interactive", help="Display losses, cmax, and cost in real-time or not", required=False)
    parser.add_argument("--version", help="The version of the current run", required=True)
    parser.add_argument("--itrs", help="The number of iterations of the current run", required=True)
    args = parser.parse_args()
    print(f"Execution mode: {args.mode}...")
    _version = int(args.version)
    _itrs = int(args.itrs)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"TPU Device: {_device}...")
    _debug_mode = (args.mode == 'test')
    agents: Agents = build_agents(device=_device, version=_version, itrs=_itrs, path=args.path, interactive=to_bool(args.interactive))
    if to_bool(args.train):
            # python main.py --train=true --path=./ --mode=prod --version=1 --itrs=0 --interactive=true 
            train(version=_version, itrs=_itrs, agents=agents, path=args.path, device=_device, debug=_debug_mode)
    else:
        if to_bool(args.target):
            # python main.py --train=false --target=true --path=./ --mode=test --version=1 --itrs=0 --size=s --id=151  --interactive=false
            dataset: Dataset = Dataset(base_path=args.path)
            solve_one_instance(instance=dataset.load_one(args.size, args.id), size=args.size, agents=agents, device=_device, path=args.path, debug=_debug_mode)
        else:
            # python main.py --train=false --target=false --path=./ --mode=prod --version=1 --itrs=0  --interactive=false
            solve_all_instances(version=_version, agents=agents, device=_device, path=args.path, debug=_debug_mode)
    print("===* END OF FILE *===")