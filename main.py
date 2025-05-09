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
def solve_one_instance(instance: Instance, size: str, agents: Agents, device: str, path: str, repetitions: int=SOLVING_REPETITIONS, debug: bool=False):
    start_time = systime.time()
    best_cmax = -1.0
    best_cost = -1.0
    best_obj = -1.0
    for rep in range(repetitions):
        print(f"SOLVING INSTANCE {size}_{id} (repetition {rep+1}/{repetitions})...")
        current_cmax, current_cost = solve(instance, agents=agents, train=False, device=device, greedy=(rep==0), debug=debug)
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
        'repetitions': [repetitions],
        'computing_time': [systime.time()-start_time],
        'device_used': [device]
    })
    print(final_metrics)
    final_metrics.to_csv(path+directory.instances+'/test/'+size+'/solution_gns_'+id+'.csv', index=False)
    return instance

# Solve all instances only in inference mode
def solve_all_instances(agents: Agents, version: int, device: str, path: str, debug:bool):
    dataset: Dataset = Dataset(base_path=path)
    dataset.load_test_instances()
    for i in dataset.test_instances:
        solve_one_instance(instance=i, size=str(i.size), agents=agents, version=version, device=device, path=path, repetitions=SOLVING_REPETITIONS, debug=debug)

def build_agents(device: str, version: int, itrs: int, path: str, train: bool):
    version = version if train else version -1
    agents: Agents = Agents(device=device, path=path+directory.models+'/', version=version)
    if version >1:
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
    _version = int(args._version)
    _itrs = int(args.itrs)
    _device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"TPU Device: {_device}...")
    _debug_mode = (args.mode == 'test')
    agents: Agents = build_agents(device=_device, version=_version, itrs=_itrs, path=args.path, train=True)
    if to_bool(args.train):
            # python main.py --train=true --mode=prod --version=1 --interactive=true --path=./
            train(version=_version, itrs=_itrs, agents=agents, path=args.path, device=_device, debug=_debug_mode)
    else:
        if to_bool(args.target):
            # SOLVE ACTUAL INSTANCE: python main.py --target=true --size=xxl --id=151 --train=false --mode=test --path=./ --version=1 --itrs=0
            # TRY ON DEBUG INSTANCE: python main.py --target=true --size=d --id=debug --train=false --mode=test --path=./ --version=1 --itrs=0
            dataset: Dataset = Dataset(base_path=args.path)
            solve_one_instance(instance=dataset.load_one(args.size, args.id), size=args.size, agents=agents, version=_version, device=_device, path=args.path, repetitions=1, debug=_debug_mode)
        else:
            # python main.py --train=false --target=false --mode=prod --path=./ --version=1 --itrs=0
            solve_all_instances(verions=_version, agents=agents, device=_device, path=args.path, debug=_debug_mode)
    print("===* END OF FILE *===")