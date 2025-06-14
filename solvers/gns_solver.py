import math
import random

import torch
torch.autograd.set_detect_anomaly(True)
from torch import Tensor
from torch.nn import Module

from conf import * 
from tools.common import debug_printer, objective_value
from tools.tensors import tensors_to_probs

from model.replay_memory import Tree, Action, HistoricalState
from model.queue import Queue
from model.instance import Instance
from model.graph import GraphInstance, State, NO, YES
from model.agent import Agents

from translators.instance2graph_translator import translate

# ############################################
# =*= GNS SOLVER FOR ONE SPECIFIC INSTANCE =*=
# ############################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

DEBUG_PRINT: callable = None

# ######################################
# =*= I. SEARCH FOR FEASIBLE ACTIONS =*=
# ######################################

# Check if an item can or must be outsourced
def can_or_must_outsource_item(instance: Instance, graph: GraphInstance, item_id: int):
    actions = []
    p, e = graph.items_g2i[item_id]
    if graph.item(item_id, 'can_be_outsourced')==YES:
        need_to_be_outsourced = False
        for o in instance.loop_item_operations(p,e):
            for rt in instance.required_rt(p, o):
                if not instance.resources_by_type(rt):
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
def get_outourcing_actions(Q: Queue, instance: Instance, graph: GraphInstance):
    actions = []
    for item_id in Q.item_queue:
        actions.extend(can_or_must_outsource_item(instance, graph, item_id))
    return actions

# Search possible material use and scheduling actions
def get_scheduling_and_material_use_actions(Q: Queue, instance: Instance, graph: GraphInstance, required_types_of_resources: list[list[list[int]]], required_types_of_materials: list[list[list[int]]]):
    scheduling_actions = []
    material_use_actions = []
    scheduling_execution_times: list[int] = []
    material_execution_times: list[int] = []
    should_start_with_scheduling: bool = True
    for operation_id in Q.operation_queue:
        p, o = graph.operations_g2i[operation_id]
        available_time = next_possible_time(instance, graph.operation(operation_id, 'available_time'), p, o) 
        first_possible_execution_time = available_time
        scheduling_sync_actions = []
        material_sync_actions = []
        _op_has_no_scheduling: bool = True
        if graph.operation(operation_id, 'remaining_resources')>0: # 1. Try for scheduling (and check for sync)
            for rt in required_types_of_resources[p][o]:
                _earliest_possible_for_this_RT = -1
                for r in graph.res_by_types[rt]:
                    _op_has_no_scheduling          = False
                    res_id                         = graph.resources_i2g[r]
                    setup_time                     = compute_setup_time(instance, graph, operation_id, res_id)
                    res_ready_time                 = graph.resource(res_id, 'available_time') + setup_time
                    scaled_res_ready_time          = next_possible_time(instance, res_ready_time, p, o)
                    _earliest_possible_for_this_RT = scaled_res_ready_time if _earliest_possible_for_this_RT<0 else min(scaled_res_ready_time, _earliest_possible_for_this_RT)
                    graph.update_need_for_resource(operation_id, res_id, [('setup_time', setup_time)])
                    if not instance.simultaneous[p][o]:
                        scheduling_actions.append((operation_id, res_id))
                        scheduling_execution_times.append(max(scaled_res_ready_time, available_time))
                    else:
                        scheduling_sync_actions.append((operation_id, res_id))
                first_possible_execution_time = max(first_possible_execution_time, _earliest_possible_for_this_RT) # for sync ops only
        if graph.operation(operation_id, 'remaining_materials')>0: # 2. Try for material use
            for rt in required_types_of_materials[p][o]:
                _earliest_possible_for_this_M = -1
                for m in graph.res_by_types[rt]:
                    mat_id = graph.materials_i2g[m]
                    mat_possible_time              = available_time if graph.material(mat_id, 'remaining_init_quantity') >= instance.quantity_needed[m][p][o] else max(instance.purchase_time[m], available_time)
                    scaled_mat_possible_time       = next_possible_time(instance, mat_possible_time, p, o)
                    _earliest_possible_for_this_M  = scaled_mat_possible_time if _earliest_possible_for_this_M<0 else min(scaled_mat_possible_time, _earliest_possible_for_this_M)
                    if not instance.simultaneous[p][o]:
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
def get_feasible_actions(Q: Queue, instance: Instance, graph: GraphInstance, required_types_of_resources: list[list[list[int]]], required_types_of_materials: list[list[list[int]]]):
    actions = [] if not Q.item_queue else get_outourcing_actions(Q, instance, graph)
    type = OUTSOURCING
    if not actions:
        actions, execution_times, type = get_scheduling_and_material_use_actions(Q, instance, graph, required_types_of_resources, required_types_of_materials)
    else:
        execution_times: list[int] = [0] * len(actions)
    return actions, type, execution_times

# #################################
# =*= II. APPLY A DECISION MADE =*=
# #################################

# Outsource item and children (reccursive down to the leaves)
def outsource_item(Q: Queue, graph: GraphInstance, instance: Instance, item_id: int, required_types_of_resources: list[list[list[int]]], required_types_of_materials: list[list[list[int]]], enforce_time: bool=False, outsourcing_time: int=-1):
    p, e = graph.items_g2i[item_id]
    cost = graph.item(item_id, 'outsourcing_cost')
    outsourcing_start_time = outsourcing_time if enforce_time else graph.item(item_id, 'start_time')
    for child in graph.direct_children[p][e]:
        _, child_end_time, child_cost = outsource_item(Q, graph, instance, graph.items_i2g[p][child], required_types_of_resources, required_types_of_materials, enforce_time=True, outsourcing_time=outsourcing_start_time)
        cost += child_cost
        outsourcing_start_time = max(outsourcing_start_time, child_end_time)
    end_date = outsourcing_start_time + instance.outsourcing_time[p][e]
    graph.update_item(item_id, [
        ('can_be_outsourced', NO),
        ('outsourced', YES),
        ('remaining_time', 0.0),
        ('children_time', 0.0),
        ('start_time', outsourcing_start_time),
        ('end_time', end_date)])
    for o in instance.loop_item_operations(p,e):
        op_id = graph.operations_i2g[p][o]
        if op_id in Q.operation_queue:
            Q.remove_operation(op_id)
        graph.update_operation(op_id, [
            ('remaining_resources', 0.0),
            ('remaining_materials', 0.0),
            ('remaining_time', 0.0)]) 
        for rt in required_types_of_resources[p][o] + required_types_of_materials[p][o]:
            for r in graph.res_by_types[rt]:
                if instance.finite_capacity[r]:
                    res_id = graph.resources_i2g[r]
                    graph.del_need_for_resource(op_id, res_id)
                    graph.inc_resource(res_id, [('remaining_operations', -1)])
                else:
                    mat_id = graph.materials_i2g[r]
                    quantity_needed = graph.need_for_material(op_id, mat_id, 'quantity_needed')
                    graph.del_need_for_material(op_id, mat_id)
                    graph.inc_material(mat_id, [('remaining_demand', -1 * quantity_needed)])
    return outsourcing_start_time, end_date, cost

# Apply an outsourcing decision to the direct parent
def apply_outsourcing_to_direct_parent(Q: Queue, instance: Instance, graph: GraphInstance, previous_operations: list, p: int, e: int, end_date: int):
    for ancestor in graph.ancesors[p][e]:
        ancestor_id = graph.items_i2g[p][ancestor]
        graph.update_item(ancestor_id, [
            ('children_time', graph.item(ancestor_id, 'children_time')-(graph.approximate_design_load[p][e]+graph.approximate_physical_load[p][e]))])
    _parent = graph.direct_parent[p][e]
    for o in graph.first_physical_operations[p][_parent]:
        next_good_to_go: bool = True
        _t = next_possible_time(instance, end_date, p, o)
        graph.update_operation(graph.operations_i2g[p][o], [('available_time', _t)], maxx=True)
        for previous in previous_operations[p][o]:
            if not graph.is_operation_complete(graph.operations_i2g[p][previous]):
                DEBUG_PRINT(f"\t >> Cannot open parent' first physical operation ({p},{o}) due to ({p},{previous}) not finished! Move at least to {_t}...")
                next_good_to_go = False
                break
        if next_good_to_go:
            DEBUG_PRINT(f"\t >> Opening first physical operation ({p},{o}) of parent {_parent} at {_t}!")
            Q.add_operation(graph.operations_i2g[p][o])

# Apply use material to an operation
def apply_use_material(graph: GraphInstance, instance: Instance, operation_id: int, material_id: int, required_types_of_materials:list[list[list[int]]], use_material_time: int):
    p, o              = graph.operations_g2i[operation_id]
    use_material_time = next_possible_time(instance, use_material_time, p, o)
    rt                = graph.resource_family[graph.materials_g2i[material_id]]
    quantity_needed   = graph.need_for_material(operation_id, material_id, 'quantity_needed')
    current_quantity  = graph.material(material_id, 'remaining_init_quantity')
    waiting_demand    = graph.material(material_id, 'remaining_demand') 
    graph.update_need_for_material(operation_id, material_id, [('status', YES), ('execution_time', use_material_time)])
    graph.update_material(material_id, [
        ('remaining_init_quantity', max(0, current_quantity - quantity_needed)),
        ('remaining_demand', waiting_demand - quantity_needed)])
    old_end = graph.operation(operation_id, 'end_time')
    graph.update_operation(operation_id, [
        ('remaining_materials', graph.operation(operation_id, 'remaining_materials') - 1),
        ('end_time', max(use_material_time, old_end))])
    required_types_of_materials[p][o].remove(rt)

# Schedule an operation on a resource
def schedule_operation(graph: GraphInstance, instance: Instance, operation_id: int, resource_id: int, required_types_of_resources: list[list[list[int]]], scheduling_time: int):
    processing_time = graph.need_for_resource(operation_id, resource_id, 'processing_time')
    p, o = graph.operations_g2i[operation_id]
    res_ready_time        = graph.resource(resource_id, 'available_time') + graph.need_for_resource(operation_id, resource_id, 'setup_time')
    scaled_res_ready_time = next_possible_time(instance, res_ready_time, p, o)
    scheduling_time       = max(scheduling_time, scaled_res_ready_time)
    operation_end         = next_possible_time(instance, scheduling_time + processing_time, p, o)
    e = graph.item_of_operations[p][o]
    r = graph.resources_g2i[resource_id]
    rt = graph.resource_family[r]
    estimated_processing_time = graph.operation_resource_time[p][o][rt]
    item_id = graph.items_i2g[p][e]
    graph.inc_resource(resource_id, [('remaining_operations', -1)])
    graph.update_resource(resource_id, [('available_time', operation_end)])
    graph.update_need_for_resource(operation_id, resource_id, [
        ('status', YES),
        ('start_time', scheduling_time),
        ('end_time', operation_end)])
    graph.current_operation_type[resource_id] = instance.get_operation_type(p, o)
    for d in range(instance.nb_settings):
        graph.current_design_value[resource_id][d] = instance.design_value[p][o][d]
    required_types_of_resources[p][o].remove(rt)
    for similar in graph.res_by_types[rt]:
        if similar != r:
            similar_id = graph.resources_i2g[similar]
            graph.inc_resource(similar_id, [('remaining_operations', -1)])
            graph.del_need_for_resource(operation_id, similar_id)
    graph.inc_operation(operation_id, [('remaining_resources', -1), ('remaining_time', -estimated_processing_time)])
    graph.update_operation(operation_id, [('end_time', operation_end), ('started', YES)], maxx=True)
    graph.update_item(item_id, [('start_time', scheduling_time)], minn=True)
    graph.update_item(item_id, [('end_time', operation_end)], maxx=True)
    graph.inc_item(item_id, [('remaining_time', -estimated_processing_time)])
    for ancestor in graph.ancesors[p][e]:
        ancestor_id = graph.items_i2g[p][ancestor]
        graph.inc_item(ancestor_id, [('children_time', -estimated_processing_time)])
    if not instance.is_design[p][o]:
        for child in graph.descendants[p][e]:
            graph.inc_item(graph.items_i2g[p][child], [('parents_physical_time', -estimated_processing_time)])
    return operation_end, scheduling_time, estimated_processing_time

# Also schedule other resources if the operation is simultaneous
def schedule_other_resources_if_simultaneous(instance: Instance, graph: GraphInstance, required_types_of_resources: list[list[list[int]]], required_types_of_materials:list[list[list[int]]], operation_id: int, resource_id: int, p: int, o: int, sync_time: int, operation_end: int):
    not_RT: int = graph.resource_family[graph.resources_g2i[resource_id]]
    total_ex_time = 0
    for rt in required_types_of_resources[p][o] + required_types_of_materials[p][o]:
        if rt != not_RT:
            found_suitable_r: bool = True 
            for r in graph.res_by_types[rt]:
                if instance.finite_capacity[r]:
                    other_resource_id       = graph.resources_i2g[r]
                    res_ready_time          = graph.resource(other_resource_id, 'available_time') + graph.need_for_resource(operation_id, other_resource_id, 'setup_time')
                    scaled_res_ready_time   = next_possible_time(instance, res_ready_time, p, o)
                    if scaled_res_ready_time <= sync_time:
                        found_suitable_r   = True
                        op_end, _, ex_time = schedule_operation(graph, instance, operation_id, other_resource_id, required_types_of_resources, sync_time)
                        operation_end      = max(operation_end, op_end)
                        total_ex_time     += ex_time
                        break
                else:
                    found_suitable_r = True
                    apply_use_material(graph, instance, operation_id, graph.materials_i2g[r], required_types_of_materials, sync_time)
                    break
            if not found_suitable_r:
                print("ERROR!")
    return operation_end, total_ex_time

# Try to open next operations after finishing using a resource or material
def try_to_open_next_operations(Q: Queue, graph: GraphInstance, instance: Instance, previous_operations: list[list[list[int]]], next_operations: list[list[list[int]]], operation_id: int): 
    p, o = graph.operations_g2i[operation_id]
    e = graph.item_of_operations[p][o]
    op_end_time = graph.operation(operation_id, 'end_time')
    for _next in next_operations[p][o]:
        next_good_to_go = True
        next_id = graph.operations_i2g[p][_next]
        for previous in previous_operations[p][_next]:
            if not graph.is_operation_complete(graph.operations_i2g[p][previous]):
                next_good_to_go = False
                break
        next_time = next_possible_time(instance, op_end_time, p, _next)
        graph.update_operation(next_id, [('available_time', next_time)], maxx=True)
        if next_good_to_go:
            DEBUG_PRINT(f'Enabling operation ({p},{_next}) at time {op_end_time} -> {next_time} in its own timescale...')
            Q.add_operation(next_id)
    if o in graph.last_design_operations[p][e]:
        for child in graph.direct_children[p][e]:
            child_id = graph.items_i2g[p][child]
            if instance.external[p][child]:
                DEBUG_PRINT(f'Enabling item {child_id} -> ({p},{child}) for outsourcing at (decision yet to make)...')
                Q.add_item(child_id)
            graph.update_item(child_id, [('start_time', op_end_time)], maxx=True)

# ####################################################
# =*= III. AUXILIARY FUNCTIONS: BUILD INIT OBJECTS =*=
# ####################################################

# Build fixed array of required resources per operation
def build_required_resources(i: Instance, graph: GraphInstance):
    required_types_of_resources = [[[] for _ in i.loop_operations(p)] for p in i.loop_projects()]
    required_types_of_materials = [[[] for _ in i.loop_operations(p)] for p in i.loop_projects()]
    res_by_types = [[] for _ in range(i.nb_resource_types)]
    for r in range(i.nb_resources):
        res_by_types[graph.resource_family[r]].append(r)
    for p in i.loop_projects():
        for o in i.loop_operations(p):
            for rt in i.required_rt(p, o):
                resources_of_rt = i.resources_by_type(rt)
                if resources_of_rt:
                    if i.finite_capacity[resources_of_rt[0]]:
                        required_types_of_resources[p][o].append(rt)
                    else:
                        required_types_of_materials[p][o].append(rt)
    return required_types_of_resources, required_types_of_materials, res_by_types

# Init the task and time queue
def init_queue(i: Instance, graph: GraphInstance):
    Q: Queue = Queue()
    for item_id in graph.project_heads:
        p, head = graph.items_g2i[item_id]
        for o in i.first_operations(p, head):
            Q.add_operation(graph.operations_i2g[p][o])
    return Q

# ################################
# =*= IV. EXECUTE ONE INSTANCE =*=
# ################################

# Select one action based on current policy
def select_next_action(agents: Agents, memory: Tree, actions_type: str, state: State, poss_actions: list[int], alpha: Tensor, train: bool=True, episode: int=1):
    model: Module = agents.agents[actions_type].policy
    if train:
        eps_threshold: float = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY_RATE)
        if random.random() > eps_threshold and memory.size >= BATCH_SIZE:
            Q_values: Tensor = model(state, poss_actions, alpha)
            return torch.argmax(Q_values.view(-1)).item() if random.random() <= GREED_TRAINING_RATE else torch.multinomial(tensors_to_probs(Q_values.view(-1)), 1).item()
        else:
            return random.randint(0, len(poss_actions)-1)
    else:
        with torch.no_grad():
            Q_values: Tensor = model(state, poss_actions, alpha)
            return torch.argmax(Q_values.view(-1)).item() if random.random() <= GREED_TESTING_RATE else torch.multinomial(tensors_to_probs(Q_values.view(-1)), 1).item()

# Compute setup times with current design settings and operation types of each finite-capacity resources
def compute_setup_time(instance: Instance, graph: GraphInstance, op_id: int, res_id: int):
    p, o = graph.operations_g2i[op_id]
    r = graph.resources_g2i[res_id]
    op_setup_time = 0 if (instance.get_operation_type(p, o) == graph.current_operation_type[res_id] or graph.current_operation_type[res_id]<0) else instance.operation_setup[r]
    for d in range(instance.nb_settings):
        op_setup_time += 0 if (graph.current_design_value[res_id][d] == instance.design_value[p][o][d] or graph.current_design_value[res_id][d]<0) else instance.design_setup[r][d] 
    return op_setup_time

# Search the next possible execution time with correct timescale of the operation
def next_possible_time(instance: Instance, time_to_test: int, p: int, o: int):
    scale = 60*instance.H if instance.in_days[p][o] else 60 if instance.in_hours[p][o] else 1
    if time_to_test % scale == 0:
        return time_to_test
    else:
        return ((time_to_test // scale) + 1) * scale

# Main function to solve an instance from sratch 
def solve(instance: Instance, agents: Agents, train: bool, device: str, REPLAY_MEMORY: Tree=None, episode: int=0, debug: bool=False):
    global DEBUG_PRINT
    DEBUG_PRINT                            = debug_printer(debug)
    best_cmax: int                         = -1
    best_cost: int                         = -1
    alpha: Tensor                          = torch.tensor([1.0], device=device)
    nb_repetitions: int                    = TRAIN_RETRY if train else TEST_RETRY
    retry: int                             = 1
    banned_step: int                       = -1
    past_decision_made: list[int]          = []
    past_nb_decisions: list[int]           = []
    past_decision_type: list[int]          = []
    while retry <= nb_repetitions:
        graph, previous_operations, next_operations = translate(i=instance, device=device)
        required_types_of_resources, required_types_of_materials, graph.res_by_types = build_required_resources(instance, graph)
        DEBUG_PRINT(f"Init Cmax: {graph.lb_Cmax}->{graph.ub_Cmax} - Init cost: {graph.lb_cost}$ - Max cost: {graph.ub_cost}$")
        current_cmax: int        = 0
        current_cost: int        = 0
        Q: Queue                 = init_queue(instance, graph)
        step: int                = 0
        decision_made: list[int] = []
        nb_decisions: list[int]  = []
        decision_type: list[int] = []
        if train:
            REPLAY_MEMORY.init_tree(alpha, graph.lb_Cmax, graph.lb_cost, graph.ub_Cmax, graph.ub_cost)
            _LOCAL_ACTION_TREE: Action = None
            _last_action: Action = None
        while not Q.done():
            workload_removed: int = 0
            poss_actions, actions_type, execution_times = get_feasible_actions(Q, instance, graph, required_types_of_resources, required_types_of_materials)
            DEBUG_PRINT(f"Current possible {ACTIONS_NAMES[actions_type]} actions: {poss_actions} at times: {execution_times}...")
            state: State = graph.to_state(device=device)
            if train:
                state_before_action: HistoricalState = HistoricalState(REPLAY_MEMORY, state, poss_actions, current_cmax, current_cost, _last_action)
                if REPLAY_MEMORY.init_state is None:
                    REPLAY_MEMORY.init_state = state_before_action
            if step < banned_step: # (1/4) forced to remake the previous decision
                idx: int = past_decision_made[step]
            elif step == banned_step:
                if actions_type == OUTSOURCING: # (2/4) switch the banned outsourcing decision to foce a change
                    item_id, past_choice = poss_actions[past_decision_made[step]]
                    new_choice = YES if past_choice == NO else NO
                    try:
                        target = (item_id, new_choice)
                        idx = poss_actions.index(target) if target in poss_actions else random.choice(range(len(poss_actions)))
                    except ValueError:
                        print("*** error outsourcing idx does not exist....")
                        idx = random.choice(range(len(poss_actions)))
                else: # (3/4) remove the banned scheduling decision to foce a change
                    banned: int = past_decision_made[step]
                    poss_actions_without_banned = poss_actions[:banned] + poss_actions[banned+1:]
                    _i = select_next_action(agents, REPLAY_MEMORY, actions_type, state, poss_actions_without_banned, alpha, train, episode)
                    idx = _i if _i < banned else _i + 1
            else: # (4/4) take a brand new decision freely
                idx = select_next_action(agents, REPLAY_MEMORY, actions_type, state, poss_actions, alpha, train, episode)
            decision_made.append(idx)
            nb_decisions.append(len(poss_actions))
            decision_type.append(actions_type)
            if actions_type == OUTSOURCING: # Outsourcing action
                item_id, outsourcing_choice = poss_actions[idx]
                target = item_id
                value = outsourcing_choice
                p, e = graph.items_g2i[item_id]
                if outsourcing_choice == YES:
                    _outsourcing_time, _end_date, _price = outsource_item(Q, graph, instance, item_id, required_types_of_resources, required_types_of_materials, enforce_time=False)
                    apply_outsourcing_to_direct_parent(Q, instance, graph, previous_operations, p, e, _end_date)
                    current_cmax = max(current_cmax, _end_date)
                    current_cost = current_cost + _price
                    Q.remove_item(item_id)
                    workload_removed = graph.approximate_item_local_time_with_children[p][e] - graph.outsourced_item_time_with_children[p][e]
                    DEBUG_PRINT(f"Outsourcing item {item_id} -> ({p},{e}) [start={_outsourcing_time}, end={_end_date}]...")
                else:
                    workload_removed = instance.outsourcing_time[p][e] - graph.approximate_item_local_time[p][e]
                    Q.remove_item(item_id)
                    graph.update_item(item_id, [('outsourced', NO), ('can_be_outsourced', NO)])
                    DEBUG_PRINT(f"Producing item {item_id} -> ({p},{e}) locally...")
            elif actions_type == SCHEDULING: # scheduling action
                operation_id, resource_id = poss_actions[idx]
                target = operation_id
                value = resource_id
                p, o = graph.operations_g2i[operation_id]
                DEBUG_PRINT(f"Scheduling: operation {operation_id} -> ({p},{o}) on resource {graph.resources_g2i[resource_id]} at time {execution_times[idx]}...")     
                _operation_end, _actual_scheduling_time, workload_removed = schedule_operation(graph, instance, operation_id, resource_id, required_types_of_resources, execution_times[idx])
                if instance.simultaneous[p][o]:
                    DEBUG_PRINT("\t >> Simulatenous...")
                    _operation_end, total_execution_time = schedule_other_resources_if_simultaneous(instance, graph, required_types_of_resources, required_types_of_materials, operation_id, resource_id, p, o, _actual_scheduling_time, _operation_end)
                    workload_removed += total_execution_time
                if graph.is_operation_complete(operation_id):
                    Q.remove_operation(operation_id)
                    try_to_open_next_operations(Q, graph, instance, previous_operations, next_operations, operation_id)
                DEBUG_PRINT(f"End of scheduling at time {_operation_end}...")
                current_cmax = max(current_cmax, _operation_end)
            else: # Material use action
                operation_id, material_id = poss_actions[idx]
                target = operation_id
                value = material_id
                p, o = graph.operations_g2i[operation_id]
                DEBUG_PRINT(f"Material use: operation {operation_id} -> ({p},{o}) on material {graph.materials_g2i[material_id]} at time {execution_times[idx]}...")  
                apply_use_material(graph, instance, operation_id, material_id, required_types_of_materials, execution_times[idx])
                if graph.is_operation_complete(operation_id):
                    Q.remove_operation(operation_id)
                    try_to_open_next_operations(Q, graph, instance, previous_operations, next_operations, operation_id)
                current_cmax = max(current_cmax, execution_times[idx])
            if train:
                _last_action = Action(idx, actions_type, target, value, workload_removed, state_before_action if _LOCAL_ACTION_TREE is not None else None)
                if _LOCAL_ACTION_TREE is None:
                    _LOCAL_ACTION_TREE = _last_action
            step += 1
        if train:
            _last_action.next_state = HistoricalState(REPLAY_MEMORY, graph.to_state(device=device), [], current_cmax, current_cost, _last_action)
            REPLAY_MEMORY.add_or_update_action(_LOCAL_ACTION_TREE, final_makespan=current_cmax, final_cost=current_cost, need_rewards=True, device=device)

        if objective_value(current_cmax, current_cost, instance.w_makespan) <= objective_value(best_cmax, best_cost, instance.w_makespan) or best_cmax < 0:
            best_cmax           = current_cmax
            best_cost           = current_cost
            nb_repetitions     += 1
            past_decision_made  = decision_made
            past_nb_decisions   = nb_decisions
            past_decision_type  = decision_type

        banned_step +=1
        while banned_step < len(past_decision_made) and (past_nb_decisions[banned_step] < 2 or past_decision_type[banned_step] == MATERIAL_USE):
            banned_step +=1
        if retry < nb_repetitions:
            print(f"RETRY {retry}/{nb_repetitions}: best nb of steps = {len(past_decision_made)} - last nb of steps = {len(decision_made)} - next banned step = {banned_step}...")
        if banned_step >= len(past_decision_made) or past_nb_decisions[banned_step] < 2:
            break
        retry += 1
    return best_cmax, best_cost
