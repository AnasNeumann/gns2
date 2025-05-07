import argparse
import pickle
import os
import time as systime

import pandas as pd
import torch
torch.autograd.set_detect_anomaly(True)
from torch.optim import Adam
from tools.common import to_bool, directory

from conf import *
from solvers.gns_solver import solve
from solvers.gns_training import train as pre_train
from model.gnn import L1_EmbbedingGNN, L1_MaterialActor, L1_OutousrcingActor, L1_SchedulingActor
from model.replay_memory import Memories
from model.instance import Instance

# ###################################################################
# =*= MAIN FILE OF THE PROJECT: CHOOSE TO TRAIN OR TEST THE MODEL =*=
# ###################################################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

SOLVING_SIZES: list[str] = ['s']
def load_test_dataset(path: str, train: bool = True):
    type: str = '/train/' if train else '/test/'
    instances = []
    for size in SOLVING_SIZES:
        complete_path = path+directory.instances+type+size+'/'
        for i in os.listdir(complete_path):
            if i.endswith('.pkl'):
                file_path = os.path.join(complete_path, i)
                with open(file_path, 'rb') as file:
                    instances.append(pickle.load(file))
    print(f"End of loading {len(instances)} instances!")
    return instances

# Compute the final objective value (to compare with other solving methos)
def objective_value(cmax: int, cost: int, cmax_weight: float):
    cmax_weight = int(100 * cmax_weight)
    cost_weight = 100 - cmax_weight
    return cmax*cmax_weight + cost*cost_weight

def load_trained_models(model_path:str, run_number:int, device:str, fine_tuned: bool = False, size: str = "", id: str = "", training_stage: bool=True):
    index = str(run_number)
    last_itr: str = str(LAST_SUCCESS)
    base_name = f"{size}_{id}_" if fine_tuned else ""
    _rm_size = GNN_CONF['resource_and_material_embedding_size']
    _io_size = GNN_CONF['operation_and_item_embedding_size']
    _hidden_size = GNN_CONF['embedding_hidden_channels']
    _ac_size = GNN_CONF['actor_hidden_channels']
    shared_GNN: L1_EmbbedingGNN = L1_EmbbedingGNN(_rm_size, _io_size, _hidden_size, GNN_CONF['nb_layers'])
    shared_GNN.load_state_dict(torch.load(model_path+'/'+base_name+'gnn_weights_'+index+'_'+last_itr+'.pth', map_location=torch.device(device), weights_only=True))
    outsourcing_actor: L1_OutousrcingActor = L1_OutousrcingActor(shared_GNN, _rm_size, _io_size, _ac_size)
    scheduling_actor: L1_SchedulingActor = L1_SchedulingActor(shared_GNN, _rm_size, _io_size, _ac_size)
    material_actor: L1_MaterialActor = L1_MaterialActor(shared_GNN, _rm_size, _io_size, _ac_size)
    outsourcing_actor.load_state_dict(torch.load(model_path+'/'+base_name+'outsourcing_weights_'+index+'_'+last_itr+'.pth', map_location=torch.device(device), weights_only=True))
    scheduling_actor.load_state_dict(torch.load(model_path+'/'+base_name+'scheduling_weights_'+index+'_'+last_itr+'.pth', map_location=torch.device(device), weights_only=True))
    material_actor.load_state_dict(torch.load(model_path+'/'+base_name+'material_use_weights_'+index+'_'+last_itr+'.pth', map_location=torch.device(device), weights_only=True))
    shared_GNN = shared_GNN.to(device)
    outsourcing_actor = outsourcing_actor.to(device)
    material_actor = material_actor.to(device)
    scheduling_actor = scheduling_actor.to(device)
    outsourcing_actor.train()
    scheduling_actor.train()
    material_actor.train()
    torch.compile(outsourcing_actor)
    torch.compile(scheduling_actor)
    torch.compile(material_actor)
    if training_stage:
        optimizer = Adam(list(scheduling_actor.parameters()) + list(material_actor.parameters()) + list(outsourcing_actor.parameters()), lr=LEARNING_RATE)
        optimizer.load_state_dict(torch.load(model_path+'/'+base_name+'adam_weights_'+index+'_'+last_itr+'.pth', map_location=torch.device(device), weights_only=True))
        with open(model_path+'/'+base_name+'memory_'+index+'_'+last_itr+'.pth', 'rb') as file:
            memory: Memories = pickle.load(file)
        return outsourcing_actor, scheduling_actor, material_actor, optimizer, memory
    return outsourcing_actor, scheduling_actor, material_actor

def init_new_models(device: str, training_stage: bool=True):
    _rm_size = GNN_CONF['resource_and_material_embedding_size']
    _io_size = GNN_CONF['operation_and_item_embedding_size']
    _hidden_size = GNN_CONF['embedding_hidden_channels']
    _ac_size = GNN_CONF['actor_hidden_channels']
    shared_GNN: L1_EmbbedingGNN = L1_EmbbedingGNN(_rm_size, _io_size, _hidden_size, GNN_CONF['nb_layers'])
    outsourcing_actor: L1_OutousrcingActor = L1_OutousrcingActor(shared_GNN, _rm_size, _io_size, _ac_size)
    scheduling_actor: L1_SchedulingActor= L1_SchedulingActor(shared_GNN, _rm_size, _io_size, _ac_size)
    material_actor: L1_MaterialActor = L1_MaterialActor(shared_GNN, _rm_size, _io_size, _ac_size)
    shared_GNN = shared_GNN.to(device)
    outsourcing_actor = outsourcing_actor.to(device)
    material_actor = material_actor.to(device)
    scheduling_actor = scheduling_actor.to(device)
    outsourcing_actor.train()
    scheduling_actor.train()
    material_actor.train()
    torch.compile(outsourcing_actor)
    torch.compile(scheduling_actor)
    torch.compile(material_actor)
    if training_stage:
        scheduling_optimizer = Adam(list(scheduling_actor.parameters()), lr=LEARNING_RATE)
        material_optimizer = Adam(list(material_actor.parameters()), lr=LEARNING_RATE)
        outsourcing_optimizer = Adam(list(outsourcing_actor.parameters()), lr=LEARNING_RATE)
        memory: Memories = Memories()
        return outsourcing_actor, scheduling_actor, material_actor, outsourcing_optimizer, scheduling_optimizer, material_optimizer, memory
    return outsourcing_actor, scheduling_actor, material_actor

# Pre-train networks on all instances
def train(run_number: int, oustourcing_agent: L1_OutousrcingActor, scheduling_agent: L1_SchedulingActor, material_actor: L1_MaterialActor, outsourcing_optimizer: Adam, scheduling_optimizer: Adam, material_optimizer: Adam, memory: Memories, device: str, path: str):
    print("Pre-training models with e-greedy DQN (on several instances)...")
    pre_train(oustourcing_agent=oustourcing_agent, scheduling_agent=scheduling_agent, material_actor=material_actor, outsourcing_optimizer=outsourcing_optimizer, scheduling_optimizer=scheduling_optimizer, material_optimizer=material_optimizer, memory=memory, path=path, solve_function=solve, device=device, run_number=run_number)
    
# Solve the target instance (size, id) only using inference
def solve_one_instance(id: str, size: str, oustourcing_agent: L1_OutousrcingActor, scheduling_agent: L1_SchedulingActor, material_actor: L1_MaterialActor, run_number: int, device: str, path: str, repetitions: int=SOLVING_REPETITIONS):
    target_instance: Instance = load_test_dataset(path+directory.instances+'/test/'+size+'/instance_'+id+'.pkl')
    start_time = systime.time()
    best_cmax = -1.0
    best_cost = -1.0
    best_obj = -1.0
    for rep in range(repetitions):
        print(f"SOLVING INSTANCE {size}_{id} (repetition {rep+1}/{repetitions})...")
        graph, current_cmax, current_cost = solve(target_instance, oustourcing_agent=oustourcing_agent, scheduling_agent=scheduling_agent, material_actor=material_actor, train=False, device=device, greedy=(rep==0))
        _obj = objective_value(current_cmax, current_cost, target_instance.w_makespan)/100
        if best_obj < 0 or _obj < best_obj:
            best_obj = _obj
            best_cmax = current_cmax
            best_cost = current_cost
    final_metrics = pd.DataFrame({
        'index': [target_instance.id],
        'value': [best_obj],
        'cmax': [best_cmax],
        'cost': [best_cost],
        'repetitions': [repetitions],
        'computing_time': [systime.time()-start_time],
        'device_used': [device]
    })
    print(final_metrics)
    final_metrics.to_csv(path+directory.instances+'/test/'+size+'/solution_gns_'+id+'.csv', index=False)
    with open(directory.solutions+'/'+size+'/gns_'+str(run_number)+'_graph_'+id+'.pkl', 'wb') as f:
            pickle.dump(graph, f)
    return target_instance

# Solve all instances only in inference mode
def solve_all_instances(oustourcing_agent: L1_OutousrcingActor, scheduling_agent: L1_SchedulingActor, material_actor: L1_MaterialActor, run_number: int, device: str, path: str):
    instances: list[Instance] = load_test_dataset(path=path, train=False)
    for i in instances:
        if (i.size, i.id):
            solve_one_instance(id=str(i.id), size=str(i.size), oustourcing_agent=oustourcing_agent, scheduling_agent=scheduling_agent, material_actor=material_actor, run_number=run_number, device=device, path=path, repetitions=SOLVING_REPETITIONS)

def agents_ready(device: str, run_number: int, path: str, train: bool):
    first = (run_number<=1)
    version = run_number if train else run_number -1
    return init_new_models(device=device, training_stage=train) if first else load_trained_models(model_path=path+directory.models, run_number=version, device=device, training_stage=train)

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
    parser.add_argument("--number", help="The number of the current run", required=True)
    args = parser.parse_args()
    print(f"Execution mode: {args.mode}...")
    _run_number = int(args.number)
    _device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"TPU Device: {_device}...")
    if to_bool(args.train):
            # python gns_solver.py --train=true --mode=prod --number=1 --interactive=true --path=./
            oustourcing_agent, scheduling_agent, material_actor, outsourcing_optimizer, scheduling_optimizer, material_optimizer, memory = agents_ready(device=_device, run_number=_run_number, path=args.path, train=True)
            train(run_number=_run_number, oustourcing_agent=oustourcing_agent, scheduling_agent=scheduling_agent, material_actor=material_actor, outsourcing_optimizer=outsourcing_optimizer, scheduling_optimizer=scheduling_optimizer, material_optimizer=material_optimizer, memory=memory, path=args.path, device=_device)
    else:
        _debug_mode = (args.mode == 'test')
        oustourcing_agent, scheduling_agent, material_actor = agents_ready(device=_device, run_number=_run_number, path=args.path, train=False)
        if to_bool(args.target):
            # SOLVE ACTUAL INSTANCE: python gns_solver.py --target=true --size=xxl --id=151 --train=false --mode=test --path=./ --number=1
            # TRY ON DEBUG INSTANCE: python gns_solver.py --target=true --size=d --id=debug --train=false --mode=test --path=./ --number=1
            i, s = solve_one_instance(id=args.id, size=args.size, oustourcing_agent=oustourcing_agent, scheduling_agent=scheduling_agent, material_actor=material_actor, run_number=args.number, device=_device, path=args.path, repetitions=1, debug=_debug_mode)
        else:
            # python gns_solver.py --train=false --target=false --mode=prod --path=./ --number=1
            solve_all_instances(run_number=args.number, oustourcing_agent=oustourcing_agent, scheduling_agent=scheduling_agent, material_actor=material_actor, device=_device, path=args.path, debug=_debug_mode)
    print("===* END OF FILE *===")