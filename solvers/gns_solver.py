import math
import random

import torch
torch.autograd.set_detect_anomaly(True)
from torch import Tensor
from torch.nn import Module

from conf import * 
from tools.common import debug_printer
from tools.tensors import tensors_to_probs

from model.replay_memory import Tree, Action, HistoricalState
from model.instance import Instance
from model.graph import State, NO, YES
from model.agent import Agents
from model.environment import Environment
from translators.instance2graph_translator import translate

# ############################################
# =*= GNS SOLVER FOR ONE SPECIFIC INSTANCE =*=
# ############################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

DEBUG_PRINT: callable = None

# Compute setup times with current design settings and operation types of each finite-capacity resources
def compute_setup_time(env: Environment, op_id: int, res_id: int):
    p, o = env.graph.operations_g2i[op_id]
    r = env.graph.resources_g2i[res_id]
    op_setup_time = 0 if (env.i.get_operation_type(p, o) == env.graph.current_operation_type[res_id] or env.graph.current_operation_type[res_id]<0) else env.i.operation_setup[r]
    for d in range(env.i.nb_settings):
        op_setup_time += 0 if (env.graph.current_design_value[res_id][d] == env.i.design_value[p][o][d] or env.graph.current_design_value[res_id][d]<0) else env.i.design_setup[r][d] 
    return op_setup_time

# Search the next possible execution time with correct timescale of the operation
def next_possible_time(instance: Instance, time_to_test: int, p: int, o: int):
    scale = 60*instance.H if instance.in_days[p][o] else 60 if instance.in_hours[p][o] else 1
    if time_to_test % scale == 0:
        return time_to_test
    else:
        return ((time_to_test // scale) + 1) * scale

# ######################################
# =*= I. SEARCH FOR FEASIBLE ACTIONS =*=
# ######################################

# Check if an item can or must be outsourced
def can_or_must_outsource_item(env: Environment, item_id: int):
    actions = []
    p, e = env.graph.items_g2i[item_id]
    if env.graph.item(item_id, 'can_be_outsourced')==YES:
        need_to_be_outsourced = False
        for o in env.i.loop_item_operations(p,e):
            for rt in env.i.required_rt(p, o):
                if not env.i.resources_by_type(rt):
                    DEBUG_PRINT(f"Unavailable resourced {rt} found, Item {item_id} must be outsourced!")
                    need_to_be_outsourced = True
                    break
            if need_to_be_outsourced:
                break
        if need_to_be_outsourced:
            actions.append((item_id, YES))
        else:
            actions.extend([(item_id, YES), (item_id, NO)])
    return actions

# Search possible outsourcing actions
def get_outourcing_actions(env: Environment):
    actions = []
    for item_id in env.Q.item_queue:
        actions.extend(can_or_must_outsource_item(env, item_id))
    return actions

# Search possible material use and scheduling actions
def get_scheduling_and_material_use_actions(env: Environment):
    scheduling_actions = []
    material_use_actions = []
    scheduling_execution_times: list[int] = []
    material_execution_times: list[int] = []
    should_start_with_scheduling: bool = True
    for operation_id in env.Q.operation_queue:
        p, o = env.graph.operations_g2i[operation_id]
        available_time = next_possible_time(env.i, env.graph.operation(operation_id, 'available_time'), p, o) 
        first_possible_execution_time = available_time
        scheduling_sync_actions = []
        material_sync_actions = []
        _op_has_no_scheduling: bool = True
        if env.graph.operation(operation_id, 'remaining_resources')>0: # 1. Try for scheduling (and check for sync)
            for rt in env.graph.remaining_types_of_resources[p][o]:
                _earliest_possible_for_this_RT = -1
                for r in env.graph.res_by_types[rt]:
                    _op_has_no_scheduling          = False
                    res_id                         = env.graph.resources_i2g[r]
                    setup_time                     = compute_setup_time(env, operation_id, res_id)
                    res_ready_time                 = env.graph.resource(res_id, 'available_time') + setup_time
                    scaled_res_ready_time          = next_possible_time(env.i, res_ready_time, p, o)
                    _earliest_possible_for_this_RT = scaled_res_ready_time if _earliest_possible_for_this_RT<0 else min(scaled_res_ready_time, _earliest_possible_for_this_RT)
                    env.graph.update_need_for_resource(operation_id, res_id, [('setup_time', setup_time)])
                    if not env.i.simultaneous[p][o]:
                        scheduling_actions.append((operation_id, res_id))
                        scheduling_execution_times.append(max(scaled_res_ready_time, available_time))
                    else:
                        scheduling_sync_actions.append((operation_id, res_id))
                first_possible_execution_time = max(first_possible_execution_time, _earliest_possible_for_this_RT) # for sync ops only
        if env.graph.operation(operation_id, 'remaining_materials')>0: # 2. Try for material use
            for rt in env.graph.remaining_types_of_materials[p][o]:
                _earliest_possible_for_this_M = -1
                for m in env.graph.res_by_types[rt]:
                    mat_id = env.graph.materials_i2g[m]
                    mat_possible_time              = available_time if env.graph.material(mat_id, 'remaining_init_quantity') >= env.i.quantity_needed[m][p][o] else max(env.i.purchase_time[m], available_time)
                    scaled_mat_possible_time       = next_possible_time(env.i, mat_possible_time, p, o)
                    _earliest_possible_for_this_M  = scaled_mat_possible_time if _earliest_possible_for_this_M<0 else min(scaled_mat_possible_time, _earliest_possible_for_this_M)
                    if not env.i.simultaneous[p][o]:
                        if _op_has_no_scheduling:
                            should_start_with_scheduling = False
                        material_use_actions.append((operation_id, mat_id))
                        material_execution_times.append(max(scaled_mat_possible_time, available_time))
                    else:
                        material_sync_actions.append((operation_id, mat_id))
                    first_possible_execution_time = max(first_possible_execution_time, _earliest_possible_for_this_M) # for sync ops only
        if scheduling_sync_actions:
                scheduling_actions.extend(scheduling_sync_actions)
                scheduling_execution_times.extend([first_possible_execution_time]*len(scheduling_sync_actions))
        if material_sync_actions:
                material_use_actions.extend(material_sync_actions)
                material_execution_times.extend([first_possible_execution_time]*len(scheduling_sync_actions))
    if (scheduling_actions and should_start_with_scheduling) or not material_use_actions:
        return scheduling_actions, scheduling_execution_times, SCHEDULING
    return material_use_actions, material_execution_times, MATERIAL_USE

# Search next possible actions with priority between decision spaces (outsourcing >> scheduling >> material use)
def get_feasible_actions(env: Environment):
    actions = [] if not env.Q.item_queue else get_outourcing_actions(env)
    type    = OUTSOURCING
    if not actions:
        actions, execution_times, type = get_scheduling_and_material_use_actions(env)
    else:
        execution_times: list[int] = [0] * len(actions)
    return actions, type, execution_times

# ######################################################
# =*= II. SELECT ONE OF THE FEASIBLE DECISIONS FOUND =*=
# ######################################################

# Select one action based on current policy
def select_one_action(agents: Agents, memory: Tree, actions_type: str, state: State, poss_actions: list[int], alpha: Tensor, train: bool=True, episode: int=1, greedy: bool=True) -> int:
    model: Module = agents.agents[actions_type].policy
    if train:
        eps_threshold: float = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / (EPS_DECAY_RATE * episode))
        if random.random() > eps_threshold and memory.size >= BATCH_SIZE:
            Q_values: Tensor = model(state, poss_actions, alpha)
            return torch.argmax(Q_values.view(-1)).item() if greedy else torch.multinomial(tensors_to_probs(Q_values.view(-1)), 1).item()
        else:
            return random.randint(0, len(poss_actions)-1)
    else:
        with torch.no_grad():
            Q_values: Tensor = model(state, poss_actions, alpha)
            return torch.argmax(Q_values.view(-1)).item() if greedy else torch.multinomial(tensors_to_probs(Q_values.view(-1)), 1).item()

# Update the environment with the selected action
def search_and_select_next_action(env: Environment, agents: Agents, greedy: bool, train: bool, REPLAY_MEMORY: Tree, episode: int, device: str, ban: bool) -> State:
    poss_actions, actions_type, execution_times = get_feasible_actions(env)
    features: State = env.graph.to_state(device=device)
    if ban: # already tried before (forbidden action)
        del poss_actions[env.action_id]
        del execution_times[env.action_id]
    DEBUG_PRINT(f"Current possible {ACTIONS_NAMES[actions_type]} actions (no ban): {poss_actions} at times: {execution_times}...")
    idx: int = select_one_action(agents, REPLAY_MEMORY, actions_type, features, poss_actions, env.alpha, train, episode, greedy)
    env.action_found(poss_actions, actions_type, idx, execution_times[idx])
    return features

# ###################################
# =*= III. APPLY A DECISION MADE =*=
# ###################################

# Outsource item and children (reccursive down to the leaves)
def outsource_item(env: Environment, item_id: int, enforce_time: bool=False, outsourcing_time: int=-1):
    p, e = env.graph.items_g2i[item_id]
    cost = env.graph.item(item_id, 'outsourcing_cost')
    outsourcing_start_time = outsourcing_time if enforce_time else env.graph.item(item_id, 'start_time')
    for child in env.graph.direct_children[p][e]:
        _, child_end_time, child_cost = outsource_item(env, env.graph.items_i2g[p][child], enforce_time=True, outsourcing_time=outsourcing_start_time)
        cost += child_cost
        outsourcing_start_time = max(outsourcing_start_time, child_end_time)
    end_date = outsourcing_start_time + env.i.outsourcing_time[p][e]
    env.graph.update_item(item_id, [
        ('can_be_outsourced', NO),
        ('outsourced', YES),
        ('remaining_time', 0.0),
        ('children_time', 0.0),
        ('start_time', outsourcing_start_time),
        ('end_time', end_date)])
    for o in env.i.loop_item_operations(p,e):
        op_id = env.graph.operations_i2g[p][o]
        if op_id in env.Q.operation_queue:
            env.Q.remove_operation(op_id)
        env.graph.update_operation(op_id, [
            ('remaining_resources', 0.0),
            ('remaining_materials', 0.0),
            ('remaining_time', 0.0)]) 
        for rt in env.graph.remaining_types_of_resources[p][o] + env.graph.remaining_types_of_materials[p][o]:
            for r in env.graph.res_by_types[rt]:
                if env.i.finite_capacity[r]:
                    res_id = env.graph.resources_i2g[r]
                    env.graph.del_need_for_resource(op_id, res_id)
                    env.graph.inc_resource(res_id, [('remaining_operations', -1)])
                else:
                    mat_id = env.graph.materials_i2g[r]
                    quantity_needed = env.graph.need_for_material(op_id, mat_id, 'quantity_needed')
                    env.graph.del_need_for_material(op_id, mat_id)
                    env.graph.inc_material(mat_id, [('remaining_demand', -1 * quantity_needed)])
    return outsourcing_start_time, end_date, cost

# Apply an outsourcing decision to the direct parent
def apply_outsourcing_to_direct_parent(env: Environment, p: int, e: int, end_date: int):
    for ancestor in env.graph.ancesors[p][e]:
        ancestor_id = env.graph.items_i2g[p][ancestor]
        env.graph.update_item(ancestor_id, [
            ('children_time', env.graph.item(ancestor_id, 'children_time')-(env.graph.approximate_design_load[p][e]+env.graph.approximate_physical_load[p][e]))])
    _parent = env.graph.direct_parent[p][e]
    for o in env.graph.first_physical_operations[p][_parent]:
        next_good_to_go: bool = True
        _t = next_possible_time(env.i, end_date, p, o)
        env.graph.update_operation(env.graph.operations_i2g[p][o], [('available_time', _t)], maxx=True)
        for previous in env.graph.previous_operations[p][o]:
            if not env.graph.is_operation_complete(env.graph.operations_i2g[p][previous]):
                DEBUG_PRINT(f"\t >> Cannot open parent' first physical operation ({p},{o}) due to ({p},{previous}) not finished! Move at least to {_t}...")
                next_good_to_go = False
                break
        if next_good_to_go:
            DEBUG_PRINT(f"\t >> Opening first physical operation ({p},{o}) of parent {_parent} at {_t}!")
            env.Q.add_operation(env.graph.operations_i2g[p][o])

# Apply use material to an operation
def apply_use_material(env: Environment, operation_id: int, material_id: int, use_material_time: int):
    p, o              = env.graph.operations_g2i[operation_id]
    use_material_time = next_possible_time(env.i, use_material_time, p, o)
    rt                = env.graph.resource_family[env.graph.materials_g2i[material_id]]
    quantity_needed   = env.graph.need_for_material(operation_id, material_id, 'quantity_needed')
    current_quantity  = env.graph.material(material_id, 'remaining_init_quantity')
    waiting_demand    = env.graph.material(material_id, 'remaining_demand') 
    env.graph.update_need_for_material(operation_id, material_id, [('status', YES), ('execution_time', use_material_time)])
    env.graph.update_material(material_id, [
        ('remaining_init_quantity', max(0, current_quantity - quantity_needed)),
        ('remaining_demand', waiting_demand - quantity_needed)])
    old_end = env.graph.operation(operation_id, 'end_time')
    env.graph.update_operation(operation_id, [
        ('remaining_materials', env.graph.operation(operation_id, 'remaining_materials') - 1),
        ('end_time', max(use_material_time, old_end))])
    env.graph.remaining_types_of_materials[p][o].remove(rt)

# Schedule an operation on a resource
def schedule_operation(env: Environment, operation_id: int, resource_id: int, scheduling_time: int):
    processing_time = env.graph.need_for_resource(operation_id, resource_id, 'processing_time')
    p, o = env.graph.operations_g2i[operation_id]
    res_ready_time        = env.graph.resource(resource_id, 'available_time') + env.graph.need_for_resource(operation_id, resource_id, 'setup_time')
    scaled_res_ready_time = next_possible_time(env.i, res_ready_time, p, o)
    scheduling_time       = max(scheduling_time, scaled_res_ready_time)
    operation_end         = next_possible_time(env.i, scheduling_time + processing_time, p, o)
    e  = env.graph.item_of_operations[p][o]
    r  = env.graph.resources_g2i[resource_id]
    rt = env.graph.resource_family[r]
    estimated_processing_time = env.graph.operation_resource_time[p][o][rt]
    item_id = env.graph.items_i2g[p][e]
    env.graph.inc_resource(resource_id, [('remaining_operations', -1)])
    env.graph.update_resource(resource_id, [('available_time', operation_end)])
    env.graph.update_need_for_resource(operation_id, resource_id, [
        ('status', YES),
        ('start_time', scheduling_time),
        ('end_time', operation_end)])
    env.graph.current_operation_type[resource_id] = env.i.get_operation_type(p, o)
    for d in range(env.i.nb_settings):
        env.graph.current_design_value[resource_id][d] = env.i.design_value[p][o][d]
    env.graph.remaining_types_of_resources[p][o].remove(rt)
    for similar in env.graph.res_by_types[rt]:
        if similar != r:
            similar_id = env.graph.resources_i2g[similar]
            env.graph.inc_resource(similar_id, [('remaining_operations', -1)])
            env.graph.del_need_for_resource(operation_id, similar_id)
    env.graph.inc_operation(operation_id, [('remaining_resources', -1), ('remaining_time', -estimated_processing_time)])
    env.graph.update_operation(operation_id, [('end_time', operation_end), ('started', YES)], maxx=True)
    env.graph.update_item(item_id, [('start_time', scheduling_time)], minn=True)
    env.graph.update_item(item_id, [('end_time', operation_end)], maxx=True)
    env.graph.inc_item(item_id, [('remaining_time', -estimated_processing_time)])
    for ancestor in env.graph.ancesors[p][e]:
        ancestor_id = env.graph.items_i2g[p][ancestor]
        env.graph.inc_item(ancestor_id, [('children_time', -estimated_processing_time)])
    if not env.i.is_design[p][o]:
        for child in env.graph.descendants[p][e]:
            env.graph.inc_item(env.graph.items_i2g[p][child], [('parents_physical_time', -estimated_processing_time)])
    return operation_end, scheduling_time, estimated_processing_time

# Also schedule other resources if the operation is simultaneous
def schedule_other_resources_if_simultaneous(env: Environment, operation_id: int, resource_id: int, p: int, o: int, sync_time: int, operation_end: int):
    not_RT: int = env.graph.resource_family[env.graph.resources_g2i[resource_id]]
    total_ex_time = 0
    for rt in env.graph.remaining_types_of_resources[p][o] + env.graph.remaining_types_of_materials[p][o]:
        if rt != not_RT:
            found_suitable_r: bool = True
            for r in env.graph.res_by_types[rt]:
                if env.i.finite_capacity[r]:
                    other_resource_id       = env.graph.resources_i2g[r]
                    res_ready_time          = env.graph.resource(other_resource_id, 'available_time') + env.graph.need_for_resource(operation_id, other_resource_id, 'setup_time')
                    scaled_res_ready_time   = next_possible_time(env.i, res_ready_time, p, o)
                    if scaled_res_ready_time <= sync_time:
                        found_suitable_r   = True
                        op_end, _, ex_time = schedule_operation(env, operation_id, other_resource_id, sync_time)
                        operation_end      = max(operation_end, op_end)
                        total_ex_time     += ex_time
                        break
                else:
                    found_suitable_r = True
                    apply_use_material(env, operation_id, env.graph.materials_i2g[r], sync_time)
                    break
            if not found_suitable_r:
                print("ERROR!")
    return operation_end, total_ex_time

# Try to open next operations after finishing using a resource or material
def try_to_open_next_operations(env: Environment, operation_id: int): 
    p, o = env.graph.operations_g2i[operation_id]
    e = env.graph.item_of_operations[p][o]
    op_end_time = env.graph.operation(operation_id, 'end_time')
    for _next in env.graph.next_operations[p][o]:
        next_good_to_go = True
        next_id = env.graph.operations_i2g[p][_next]
        for previous in env.graph.previous_operations[p][_next]:
            if not env.graph.is_operation_complete(env.graph.operations_i2g[p][previous]):
                next_good_to_go = False
                break
        next_time = next_possible_time(env.i, op_end_time, p, _next)
        env.graph.update_operation(next_id, [('available_time', next_time)], maxx=True)
        if next_good_to_go:
            DEBUG_PRINT(f'Enabling operation ({p},{_next}) at time {op_end_time} -> {next_time} in its own timescale...')
            env.Q.add_operation(next_id)
    if o in env.graph.last_design_operations[p][e]:
        for child in env.graph.direct_children[p][e]:
            child_id = env.graph.items_i2g[p][child]
            if env.i.external[p][child]:
                DEBUG_PRINT(f'Enabling item {child_id} -> ({p},{child}) for outsourcing at (decision yet to make)...')
                env.Q.add_item(child_id)
            env.graph.update_item(child_id, [('start_time', op_end_time)], maxx=True)

# ##########################################################################
# =*= IV. EXECUTE ONE DECISION AND ONE COMPLETE INSTANCE (ALL DECISIONS) =*=
# ##########################################################################

# Solve one instance starting from a given graph (instance + partial solution)
def execute_one_decision(env: Environment, features: State, train: bool, REPLAY_MEMORY: Tree=None):
    workload_removed: int = 0
    if train:
        state_before_action: HistoricalState = HistoricalState(REPLAY_MEMORY, features, env.possible_actions, env.current_cmax, env.current_cost, env.get_last_action())
        if REPLAY_MEMORY.init_state is None:
            REPLAY_MEMORY.init_state = state_before_action
    if env.action_type == OUTSOURCING: # Outsourcing action
        item_id, outsourcing_choice = env.possible_actions[env.action_id]
        target = item_id
        value = outsourcing_choice
        p, e = env.graph.items_g2i[item_id]
        if outsourcing_choice == YES:
            _outsourcing_time, _end_date, _price = outsource_item(env, item_id, enforce_time=False)
            apply_outsourcing_to_direct_parent(env, p, e, _end_date)
            env.current_cmax = max(env.current_cmax, _end_date)
            env.current_cost = env.current_cost + _price
            env.Q.remove_item(item_id)
            workload_removed = env.graph.outsourced_item_time_with_children[p][e] - env.graph.approximate_item_local_time_with_children[p][e]
            DEBUG_PRINT(f"Outsourcing item {item_id} -> ({p},{e}) [start={_outsourcing_time}, end={_end_date}]...")
        else:
            workload_removed = env.graph.approximate_item_local_time[p][e] - env.i.outsourcing_time[p][e]
            env.Q.remove_item(item_id)
            env.graph.update_item(item_id, [('outsourced', NO), ('can_be_outsourced', NO)])
            DEBUG_PRINT(f"Producing item {item_id} -> ({p},{e}) locally...")
    elif env.action_type == SCHEDULING: # scheduling action
        operation_id, resource_id = env.possible_actions[env.action_id]
        target = operation_id
        value = resource_id
        p, o = env.graph.operations_g2i[operation_id]
        DEBUG_PRINT(f"Scheduling: operation {operation_id} -> ({p},{o}) on resource {env.graph.resources_g2i[resource_id]} at time {env.execution_time}...")     
        _operation_end, _actual_scheduling_time, workload_removed = schedule_operation(env, operation_id, resource_id, env.execution_time)
        if env.i.simultaneous[p][o]:
            DEBUG_PRINT("\t >> Simulatenous...")
            _operation_end, total_execution_time = schedule_other_resources_if_simultaneous(env, operation_id, resource_id, p, o, _actual_scheduling_time, _operation_end)
            workload_removed += total_execution_time
        if env.graph.is_operation_complete(operation_id):
            env.Q.remove_operation(operation_id)
            try_to_open_next_operations(env, operation_id)
        DEBUG_PRINT(f"End of scheduling at time {_operation_end}...")
        env.current_cmax = max(env.current_cmax, _operation_end)
    else: # Material use action
            operation_id, material_id = env.possible_actions[env.action_id]
            target = operation_id
            value = material_id
            p, o = env.graph.operations_g2i[operation_id]
            DEBUG_PRINT(f"Material use: operation {operation_id} -> ({p},{o}) on material {env.graph.materials_g2i[material_id]} at time {env.execution_time}...")  
            apply_use_material(env, operation_id, material_id, env.execution_time)
            if env.graph.is_operation_complete(operation_id):
                env.Q.remove_operation(operation_id)
                try_to_open_next_operations(env, operation_id)
            env.current_cmax = max(env.current_cmax, env.execution_time)
    if train:
        is_first_env: bool = env.is_first_env()
        env.previous_actions.append(Action(env.action_id, env.action_type, target, value, workload_removed, state_before_action if not is_first_env else None))

# Main function to solve an instance from sratch 
def solve(instance: Instance, agents: Agents, train: bool, device: str, greedy: bool=False, REPLAY_MEMORY: Tree=None, episode: int=0, debug: bool=False) -> Environment:
    global DEBUG_PRINT
    DEBUG_PRINT = debug_printer(debug)
    graph, lb_Cmax, ub_Cmax, lb_cost, ub_cost = translate(i=instance, device=device)
    env: Environment = Environment(instance=instance, graph=graph, lb_Cmax=lb_Cmax, ub_Cmax=ub_Cmax, lb_cost=lb_cost, ub_cost=ub_cost, alpha=torch.tensor([instance.w_makespan], device=device))
    if train:
        REPLAY_MEMORY.init_tree(env.alpha, env.lb_Cmax, env.lb_cost, env.ub_Cmax, env.ub_cost)
    DEBUG_PRINT(f"Init Cmax: {env.lb_Cmax}->{env.ub_Cmax} - Init cost: {env.lb_cost}$ - Max cost: {env.ub_cost}$")
    env.init_queue()
    best_step_envs: list[Environment] = []
    best_solution: Environment        = None
    retry: int                        = 1
    ban: bool                         = False
    starting_step: int                = 0
    limit_retries: int                = MAX_RETRY_TRAIN if train else MAX_RETRY_INF
    while retry <= limit_retries:

        # 1. Execute an instance either from scratch or starting from a step -----
        print(f"RETRY {retry}/{limit_retries} - starting from step {starting_step}...")
        step_envs: list[Environment] = best_step_envs[:starting_step].copy() if best_step_envs else []
        while not env.Q.done():
            features = search_and_select_next_action(env=env, agents=agents, greedy=greedy, train=train, REPLAY_MEMORY=REPLAY_MEMORY, episode=episode, device=device, ban=ban)
            step_envs.append(env.clone())
            execute_one_decision(env, features, train, REPLAY_MEMORY=REPLAY_MEMORY)
            ban = False
        if train:
            last_action: Action = env.get_last_action()
            last_action.next_state = HistoricalState(REPLAY_MEMORY, env.graph.to_state(device=device), [], env.current_cmax, env.current_cost, last_action)
            REPLAY_MEMORY.add_or_update_action(env.get_base_action(), final_makespan=env.current_cmax, final_cost=env.current_cost, need_rewards=True, device=device)

        # 2. Check if a new best solution is found ----------------------------
        if best_solution is None or env.obj_value() <= best_solution.obj_value():
            best_solution  = env
            best_step_envs = step_envs
            if not train:
                limit_retries += 1

        # 3. Try to reset the environment and restart -------------------------
        test: int = 0
        found: bool = False
        while test < MAX_TEST_FOR_RETRY:
            starting_step = random.randrange(len(best_step_envs)-1)
            if len(best_step_envs[starting_step].possible_actions) > 1:
                found = True
                break
            test += 1
        if not found:
            break
        env: Environment = best_step_envs[starting_step].clone()
        retry += 1
        ban = True
    return best_solution