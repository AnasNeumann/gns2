from model.instance import Instance
from model.graph import GraphInstance, ItemFeatures, OperationFeatures, ResourceFeatures, MaterialFeatures, NeedForMaterialFeatures, NeedForResourceFeatures, YES, NO
from torch import Tensor

# #####################################
# =*=  TRANSLATE INSTANCE 2 GRAPH   =*=
# model.Instance to model.GraphInstance
# #####################################
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

def build_item(i: Instance, graph: GraphInstance, p: int, e: int, head: bool, external_start: int, internal_start: int, must_be_outsourced: bool=False):
    design_mean_time, physical_mean_time = i.item_processing_time(p, e, total_load=False, max_load=False)
    design_max_time, physical_max_time   = i.item_processing_time(p, e, total_load=False, max_load=True)
    childrens = graph.descendants[p][e]
    children_time = 0
    childrens_ops = 0
    for children in childrens:
        cdt = graph.approximate_design_load[p][children]
        cpt = graph.approximate_physical_load[p][children]
        children_time += (cdt+cpt)
        for _ in i.loop_item_operations(p, e=children):
            childrens_ops += 1
    parents_physical_time = 0
    parent_physical_ops = 0
    for ancestor in graph.ancesors[p][e]:
        apt = graph.approximate_physical_load[p][ancestor]
        parents_physical_time += apt
        for parent_op in i.loop_item_operations(p, e=ancestor):
            if not i.is_design[p][parent_op]:
                parent_physical_ops += 1
    item_id = graph.add_item(p, e, ItemFeatures(
        can_be_outsourced = YES if i.external[p][e] else NO,
        outsourced = NO,
        outsourcing_cost = float(i.external_cost[p][e]) if i.external[p][e] else 0.0,
        outsourcing_time = float(i.outsourcing_time[p][e]) if i.external[p][e] else 0.0,
        remaining_time = float(graph.approximate_physical_load[p][e] + graph.approximate_design_load[p][e]),
        parents = float(len(graph.ancesors[p][e])),
        children = float(len(childrens)),
        parents_physical_time = float(parents_physical_time),
        children_time = float(children_time),
        start_time = 0.0,
        end_time = 0.0), head = head)
    start, end = i.get_operations_idx(p,e)
    physical_ops_ids: list = []
    pr_ids: list = []
    pm_ids: list = []
    for o in range(start, end):
        succs = end-(o+1)
        operation_load = i.operation_time(p,o, total_load=True)
        required_res = 0.0
        required_mat = 0.0
        for rt in i.required_rt(p, o):
            resources_of_rt = i.resources_by_type(rt)
            if resources_of_rt:
                if i.finite_capacity[resources_of_rt[0]]:
                    required_res += 1
                else:
                    required_mat += 1
            else:
                must_be_outsourced = True
        op_id = graph.add_operation(p, o, OperationFeatures(
            started = NO,
            sync = float(i.simultaneous[p][o]),
            large_timescale =float(i.in_days[p][o]),
            successors = float(parent_physical_ops + succs + (childrens_ops if i.is_design[p][o] else 0.0)),
            remaining_time = float(operation_load),
            remaining_resources = float(required_res),
            remaining_materials = float(required_mat),
            available_time = 0.0,
            lb = 0.0,
            end_time = 0.0))
        if not i.is_design[p][o]:
            physical_ops_ids.append(op_id)
        graph.add_operation_assembly(item_id, op_id)
        for rt in i.required_rt(p, o):
            for r in i.resources_by_type(rt):
                if i.finite_capacity[r]:
                    _ext =  i.execution_time[r][p][o]
                    graph.add_need_for_resources(op_id, graph.resources_i2g[r], NeedForResourceFeatures(
                        status = NO,
                        setup_time = 0.0,
                        processing_time = float(_ext),
                        start_time = 0.0,
                        end_time = 0.0))
                    if not i.is_design[p][o]:
                        pr_ids.append((op_id, graph.resources_i2g[r]))
                else:
                    graph.add_need_for_materials(op_id, graph.materials_i2g[r], NeedForMaterialFeatures(
                        status = NO,
                        execution_time = 0.0,
                        quantity_needed = float(i.quantity_needed[r][p][o])))
                    if not i.is_design[p][o]:
                        pm_ids.append((op_id, graph.materials_i2g[r]))
    _base_design_end = internal_start + design_mean_time
    _max_design_end = internal_start + design_max_time
    external_start_child = external_start if i.external[p][e] else _base_design_end
    least_childrend_end = _base_design_end
    max_childrend_end = _max_design_end
    least_children_cost = 0
    max_children_cost = 0
    for children in graph.direct_children[p][e]:
        child_id, least_end_child, max_end_child, least_child_cost, max_child_cost = build_item(i, graph, p, e=children, head=False, internal_start=external_start_child, external_start=_base_design_end, must_be_outsourced=must_be_outsourced)
        graph.add_item_assembly(item_id, child_id)
        least_childrend_end = max(least_childrend_end, least_end_child)
        max_childrend_end = max(max_childrend_end, max_end_child)
        least_children_cost += least_child_cost
        max_children_cost += max_child_cost
    least_external_end = least_childrend_end + i.outsourcing_time[p][e]
    max_external_end   = max_childrend_end + i.outsourcing_time[p][e]
    least_internal_end = least_children_cost + physical_mean_time
    max_internal_end   = max_children_cost + physical_max_time
    least_end          = least_external_end if must_be_outsourced else min(least_external_end, least_internal_end) if i.external[p][e] else least_internal_end
    max_end            = max_external_end if must_be_outsourced else max(max_external_end, max_internal_end) if i.external[p][e] else max_internal_end
    least_cost         = least_children_cost + (i.external_cost[p][e] if must_be_outsourced else 0)
    max_cost           = max_children_cost + (i.external_cost[p][e] if i.external[p][e] else 0)
    return item_id, least_end, max_end, least_cost, max_cost

def build_precedence(i: Instance, graph: GraphInstance):
    for p in i.loop_projects():
        for e in i.loop_items(p):
            start, end = i.get_operations_idx(p, e)
            parent_last_design = []
            parent_first_physical = []
            parent = graph.direct_parent[p][e]
            if parent > -1:
                parent_last_design = graph.last_design_operations[p][parent]
                parent_first_physical = graph.first_physical_operations[p][parent]
            for o in range(start, end):
                o_id = graph.operations_i2g[p][o]
                preds = i.preds_or_succs(p, e, start, end, o, design_only=False, physical_only=False, preds=True)
                for pred in preds:
                    pred_id = graph.operations_i2g[p][pred]
                    graph.add_precedence(pred_id, o_id)
                if parent > -1 and o == start:
                    for pld in parent_last_design:
                        pred_id = graph.operations_i2g[p][pld]
                        graph.add_precedence(pred_id, o_id)
                if parent > -1 and o == (end-1):
                    for pfp in parent_first_physical:
                        succ_id = graph.operations_i2g[p][pfp]
                        graph.add_precedence(o_id, succ_id)

def build_direct_access_objects(i: Instance, graph: GraphInstance):
    for p in i.loop_projects():
        graph.direct_parent.append([])
        graph.direct_children.append([])
        graph.ancesors.append([])
        graph.descendants.append([])
        graph.last_design_operations.append([])
        graph.first_physical_operations.append([])
        graph.approximate_design_load.append([])
        graph.approximate_physical_load.append([])  
        graph.operation_resource_time.append([])
        graph.item_of_operations.append([])
        graph.approximate_item_local_time.append([])
        for e in i.loop_items(p):
            adl, apl = i.item_processing_time(p, e, total_load=True)
            mean_adl, mean_apl = i.item_processing_time(p, e, total_load=False)
            graph.approximate_item_local_time[p].append(mean_adl + mean_apl)
            graph.approximate_design_load[p].append(adl)
            graph.approximate_physical_load[p].append(apl)
            graph.direct_parent[p].append(i.get_direct_parent(p, e))
            graph.ancesors[p].append(i.get_ancestors(p, e))
            graph.direct_children[p].append(i.get_children(p, e, direct=True))
            graph.descendants[p].append(i.get_children(p, e, direct=False))
            graph.last_design_operations[p].append(i.last_design_operations(p, e))
            graph.first_physical_operations[p].append(i.first_physical_operations(p, e))
        for o in i.loop_operations(p):
            graph.item_of_operations[p].append(i.get_item_of_operation(p, o))
            graph.operation_resource_time[p].append([])
            for rt in i.loop_rts():
                graph.operation_resource_time[p][o].append(i.operation_resource_time(p, o, rt, max_load=True))
    for r in i.loop_resources():
        graph.resource_family.append(i.get_resource_familly(r))
    for p in i.loop_projects():
        graph.outsourced_item_time_with_children.append([])
        graph.approximate_item_local_time_with_children.append([])
        for e in i.loop_items(p):
            external_t: int = i.outsourcing_time[p][e]
            internal_t: int = graph.approximate_item_local_time[p][e]
            for d in graph.descendants[p][e]:
                external_t += i.outsourcing_time[p][d]
                internal_t += graph.approximate_item_local_time[p][d]
            graph.outsourced_item_time_with_children[p].append(external_t)
            graph.approximate_item_local_time_with_children[p].append(internal_t)

def rec_lower_bound(i: Instance, graph: GraphInstance, p: int, o: int, start: int, next_operations: list[list[list[int]]]):
    graph.update_operation(graph.operations_i2g[p][o], [('lb', start)], maxx=True)
    for next in next_operations[p][o]:
        rec_lower_bound(i, graph, p, next, start + i.operation_time(p, o, total_load=False), next_operations)

def build_lower_bounds(i: Instance, graph: GraphInstance, next_operations: list[list[list[int]]]):
    for p in i.loop_projects():
        for o in i.first_operations(p, i.project_head(p)):
            rec_lower_bound(i, graph, p, o, 0, next_operations)

# Build fixed array of required resources per operation
def build_required_resources(i: Instance, graph: GraphInstance):
    graph.remaining_types_of_resources = [[[] for _ in i.loop_operations(p)] for p in i.loop_projects()]
    graph.remaining_types_of_materials = [[[] for _ in i.loop_operations(p)] for p in i.loop_projects()]
    graph.res_by_types = [[] for _ in range(i.nb_resource_types)]
    for r in range(i.nb_resources):
        graph.res_by_types[graph.resource_family[r]].append(r)
    for p in i.loop_projects():
        for o in i.loop_operations(p):
            for rt in i.required_rt(p, o):
                resources_of_rt = i.resources_by_type(rt)
                if resources_of_rt:
                    if i.finite_capacity[resources_of_rt[0]]:
                        graph.remaining_types_of_resources[p][o].append(rt)
                    else:
                        graph.remaining_types_of_materials[p][o].append(rt) 

def translate(i: Instance, device: str):
    graph = GraphInstance(device=device)
    build_direct_access_objects(i, graph)
    for rt in range(i.nb_resource_types):
        resources = i.resources_by_type(rt)
        operations = i.operations_by_resource_type(rt)
        res_idx = []
        for r in resources:
            if i.finite_capacity[r]:
                res_id = graph.add_resource(r, i.nb_settings, ResourceFeatures(
                    available_time = 0.0,
                    remaining_operations = float(len(operations)),
                    similar_resources = float(len(resources))))
                for other_id in res_idx:
                    graph.add_same_types(other_id, res_id)
                res_idx.append(res_id)
            else:
                remaining_quantity_needed = 0
                for p, o in operations:
                    remaining_quantity_needed += i.quantity_needed[r][p][o]
                graph.add_material(r, MaterialFeatures(
                    remaining_init_quantity = float(i.init_quantity[r]), 
                    arrival_time = float(i.purchase_time[r]), 
                    remaining_demand = float(remaining_quantity_needed)))
    graph.resources_i2g = graph.build_i2g_1D(graph.resources_g2i, i.nb_resources)
    graph.materials_i2g = graph.build_i2g_1D(graph.materials_g2i, i.nb_resources)
    lb_Cmax, ub_Cmax, lb_cost, ub_cost = 0, 0, 0, 0
    for p in i.loop_projects():
        _, project_lb_end, project_ub_end, project_lb_cost, project_ub_cost = build_item(i, graph, p, e=i.project_head(p), head=True, external_start=0, internal_start=0, must_be_outsourced=False)
        lb_Cmax = max(graph.lb_Cmax, project_lb_end)
        ub_Cmax = max(graph.ub_Cmax, project_ub_end)
        lb_cost += project_lb_cost
        ub_cost += project_ub_cost
    graph.operations_i2g = graph.build_i2g_2D(graph.operations_g2i)
    graph.items_i2g = graph.build_i2g_2D(graph.items_g2i)
    graph.add_dummy_item(device=device)
    build_precedence(i, graph)
    graph.previous_operations, graph.next_operations = i.build_next_and_previous_operations()
    build_lower_bounds(i, graph, graph.next_operations)
    build_required_resources(i, graph)
    return graph, lb_Cmax, ub_Cmax, lb_cost, ub_cost